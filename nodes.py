import json
import os
from PIL import Image, ImageOps
import numpy as np
import torch
import folder_paths
import node_helpers
import comfy.utils
from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords
from comfy_api.latest import io


def _parse_guides_json(guides_json):
    if guides_json is None:
        raise ValueError("guides_json is required")

    raw = guides_json.strip()
    if not raw:
        raise ValueError("guides_json is empty")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        if os.path.exists(raw):
            with open(raw, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise

    if isinstance(data, dict) and "guides" in data:
        guides = data["guides"]
    elif isinstance(data, list):
        guides = data
    else:
        raise ValueError("guides_json must be a list or a dict with a 'guides' key")

    cleaned = []
    for idx, item in enumerate(guides):
        if not isinstance(item, dict):
            raise ValueError(f"guide #{idx} must be an object")

        image_path = item.get("image") or item.get("image_path") or item.get("path")
        if not image_path:
            raise ValueError(f"guide #{idx} missing 'image' path")

        frame_idx = item.get("frame_idx", item.get("frame", 0))
        strength = item.get("strength", 1.0)

        cleaned.append({
            "image": image_path,
            "frame_idx": int(frame_idx),
            "strength": float(strength),
        })

    return cleaned


def _resolve_image_path(image_path):
    path = image_path.strip()
    if os.path.isabs(path) and os.path.exists(path):
        return path

    if os.path.exists(path):
        return path

    try:
        resolved = folder_paths.get_annotated_filepath(path)
        if os.path.exists(resolved):
            return resolved
    except Exception:
        pass

    raise FileNotFoundError(f"Image not found: {image_path}")


def _load_image_tensor(image_path):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    image = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(image)[None,]


_LTXV_PATCHIFIER = SymmetricPatchifier(1, start_end=True)


def _get_noise_mask(latent):
    noise_mask = latent.get("noise_mask", None)
    latent_image = latent["samples"]
    if noise_mask is None:
        batch_size, _, latent_length, _, _ = latent_image.shape
        noise_mask = torch.ones(
            (batch_size, 1, latent_length, 1, 1),
            dtype=torch.float32,
            device=latent_image.device,
        )
    else:
        noise_mask = noise_mask.clone()
    return noise_mask


def _get_keyframe_idxs(cond):
    for t in cond:
        if "keyframe_idxs" in t[1]:
            keyframe_idxs = t[1]["keyframe_idxs"]
            break
    else:
        return None, 0

    num_keyframes = torch.unique(keyframe_idxs[:, 0, :, 0]).shape[0]
    return keyframe_idxs, num_keyframes


def _add_keyframe_index(cond, frame_idx, guiding_latent, scale_factors):
    keyframe_idxs, _ = _get_keyframe_idxs(cond)
    _, latent_coords = _LTXV_PATCHIFIER.patchify(guiding_latent)
    pixel_coords = latent_to_pixel_coords(latent_coords, scale_factors, causal_fix=frame_idx == 0)
    pixel_coords[:, 0] += frame_idx

    if keyframe_idxs is None:
        keyframe_idxs = pixel_coords
    else:
        keyframe_idxs = torch.cat([keyframe_idxs, pixel_coords], dim=2)
    return node_helpers.conditioning_set_values(cond, {"keyframe_idxs": keyframe_idxs})


def _append_keyframe(positive, negative, frame_idx, latent_image, noise_mask, guiding_latent, strength, scale_factors):
    if latent_image.shape[1] != 128 or guiding_latent.shape[1] != 128:
        raise ValueError("Adding guide to a combined AV latent is not supported.")

    positive = _add_keyframe_index(positive, frame_idx, guiding_latent, scale_factors)
    negative = _add_keyframe_index(negative, frame_idx, guiding_latent, scale_factors)

    mask = torch.full(
        (noise_mask.shape[0], 1, guiding_latent.shape[2], noise_mask.shape[3], noise_mask.shape[4]),
        1.0 - strength,
        dtype=noise_mask.dtype,
        device=noise_mask.device,
    )

    latent_image = torch.cat([latent_image, guiding_latent], dim=2)
    noise_mask = torch.cat([noise_mask, mask], dim=2)
    return positive, negative, latent_image, noise_mask


def _encode_guide(vae, latent_width, latent_height, images, scale_factors):
    time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
    images = images[:(images.shape[0] - 1) // time_scale_factor * time_scale_factor + 1]
    pixels = comfy.utils.common_upscale(
        images.movedim(-1, 1),
        latent_width * width_scale_factor,
        latent_height * height_scale_factor,
        "bilinear",
        crop="disabled",
    ).movedim(1, -1)
    encode_pixels = pixels[:, :, :, :3]
    t = vae.encode(encode_pixels)
    return encode_pixels, t


def _get_latent_index(cond, latent_length, guide_length, frame_idx, scale_factors):
    time_scale_factor, _, _ = scale_factors
    # Negative frame_idx means positions before frame 0 (do not remap to end).
    if frame_idx >= 0 and guide_length > 1 and frame_idx != 0:
        frame_idx = (frame_idx - 1) // time_scale_factor * time_scale_factor + 1

    latent_idx = (frame_idx + time_scale_factor - 1) // time_scale_factor
    return frame_idx, latent_idx


class LTXVAddGuideMultiJsonFc:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "guides_json": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "add"
    CATEGORY = "LTX2"

    def add(self, positive, negative, vae, latent, guides_json):
        guides = _parse_guides_json(guides_json)

        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = _get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        for guide in guides:
            image_path = _resolve_image_path(guide["image"])
            img = _load_image_tensor(image_path)
            f_idx = guide["frame_idx"]
            strength = guide["strength"]

            image_1, t = _encode_guide(vae, latent_width, latent_height, img, scale_factors)
            frame_idx, latent_idx = _get_latent_index(positive, latent_length, len(image_1), f_idx, scale_factors)

            if latent_idx + t.shape[2] > latent_length:
                raise ValueError("Conditioning frames exceed the length of the latent sequence.")

            positive, negative, latent_image, noise_mask = _append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                t,
                strength,
                scale_factors,
            )

        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask})


class LTXVAddGuideMultiFc(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = []
        for num_guides in range(1, 21):
            guide_inputs = []
            for i in range(1, num_guides + 1):
                guide_inputs.extend([
                    io.Image.Input(f"image_{i}"),
                    io.Int.Input(
                        f"frame_idx_{i}",
                        default=0,
                        min=-9999,
                        max=9999,
                        tooltip=f"Frame index for guide {i}.",
                    ),
                    io.Float.Input(
                        f"strength_{i}",
                        default=1.0,
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        tooltip=f"Strength for guide {i}.",
                    ),
                ])
            options.append(io.DynamicCombo.Option(
                key=str(num_guides),
                inputs=guide_inputs,
            ))

        return io.Schema(
            node_id="LTXVAddGuideMultiFc",
            category="KJNodes/ltxv",
            description=(
                "Add multiple guide images at specified frame indices with strengths, "
                "uses DynamicCombo which requires ComfyUI 0.8.1 and frontend 1.33.4 or later."
            ),
            inputs=[
                io.Conditioning.Input("positive", tooltip="Positive conditioning to which guide keyframe info will be added"),
                io.Conditioning.Input("negative", tooltip="Negative conditioning to which guide keyframe info will be added"),
                io.Vae.Input("vae", tooltip="Video VAE used to encode the guide images"),
                io.Latent.Input("latent", tooltip="Video latent, guides are added to the end of this latent"),
                io.DynamicCombo.Input(
                    "num_guides",
                    options=options,
                    display_name="Number of Guides",
                    tooltip="Select how many guide images to use",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="Video latent with added guides"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, latent, num_guides) -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = _get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape

        image_keys = sorted([k for k in num_guides.keys() if k.startswith("image_")])

        for img_key in image_keys:
            i = img_key.split("_")[1]

            img = num_guides[f"image_{i}"]
            f_idx = num_guides[f"frame_idx_{i}"]
            strength = num_guides[f"strength_{i}"]

            image_1, t = _encode_guide(vae, latent_width, latent_height, img, scale_factors)

            frame_idx, latent_idx = _get_latent_index(positive, latent_length, len(image_1), f_idx, scale_factors)
            assert latent_idx + t.shape[2] <= latent_length, "Conditioning frames exceed the length of the latent sequence."

            positive, negative, latent_image, noise_mask = _append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                t,
                strength,
                scale_factors,
            )

        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

NODE_CLASS_MAPPINGS = {
    "LTXVAddGuideMultiJsonFc": LTXVAddGuideMultiJsonFc,
    "LTXVAddGuideMultiFc": LTXVAddGuideMultiFc,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVAddGuideMultiJsonFc": "LTXV Add Guide Multi (JSON) FC",
    "LTXVAddGuideMultiFc": "LTXV Add Guide Multi FC",
}
