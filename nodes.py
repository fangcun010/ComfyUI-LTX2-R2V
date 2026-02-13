import json
import os
from PIL import Image, ImageOps
import numpy as np
import torch
import folder_paths
import node_helpers
import comfy.utils
import comfy_extras.nodes_lt as nodes_lt
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
        preprocess = item.get("preprocess", True)
        preprocess_crf = item.get("preprocess_crf", 33)

        cleaned.append({
            "image": image_path,
            "frame_idx": int(frame_idx),
            "strength": float(strength),
            "preprocess": bool(preprocess),
            "preprocess_crf": int(preprocess_crf),
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


def _build_mask_ramp(guide_frames, slope_len, device, dtype):
    if guide_frames <= 1:
        return torch.ones((guide_frames,), device=device, dtype=dtype)

    slope_len = max(1, min(slope_len, guide_frames // 2))
    coeffs = torch.ones((guide_frames,), device=device, dtype=dtype)
    for i in range(slope_len):
        value = (i + 1) / slope_len
        coeffs[i] = value
        coeffs[-(i + 1)] = value
    return coeffs


def _maybe_expand_guide(image_1, guiding_latent, mask_mode, ramp_frames):
    if mask_mode != "ramp" or ramp_frames <= 1:
        return image_1, guiding_latent

    if guiding_latent.shape[2] == 1 and ramp_frames > 1:
        guiding_latent = guiding_latent.repeat(1, 1, ramp_frames, 1, 1)
        image_1 = image_1.repeat(ramp_frames, 1, 1, 1)
    return image_1, guiding_latent


def _append_keyframe(
    positive,
    negative,
    frame_idx,
    latent_image,
    noise_mask,
    guiding_latent,
    strength,
    scale_factors,
    mask_mode,
    ramp_frames,
):
    if latent_image.shape[1] != 128 or guiding_latent.shape[1] != 128:
        raise ValueError("Adding guide to a combined AV latent is not supported.")

    positive = _add_keyframe_index(positive, frame_idx, guiding_latent, scale_factors)
    negative = _add_keyframe_index(negative, frame_idx, guiding_latent, scale_factors)

    guide_frames = guiding_latent.shape[2]
    if mask_mode == "ramp":
        slope_len = max(1, min(ramp_frames // 2, guide_frames // 2))
        coeffs = _build_mask_ramp(
            guide_frames,
            slope_len,
            noise_mask.device,
            noise_mask.dtype,
        )
        coeffs = coeffs.view(1, 1, guide_frames, 1, 1)
        mask = 1.0 - (strength * coeffs)
        mask = mask.expand(
            noise_mask.shape[0],
            1,
            guide_frames,
            noise_mask.shape[3],
            noise_mask.shape[4],
        )
    else:
        mask = torch.full(
            (noise_mask.shape[0], 1, guide_frames, noise_mask.shape[3], noise_mask.shape[4]),
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


def _resize_for_preprocess(images, target_width, target_height, upscale_method):
    divisible_by = 32

    if divisible_by > 1:
        target_width = target_width - (target_width % divisible_by)
        target_height = target_height - (target_height % divisible_by)

    _, height, width, _ = images.shape

    if target_width == 0 and target_height == 0:
        target_width = width
        target_height = height
    elif target_width == 0:
        target_width = round(width * (target_height / height))
    elif target_height == 0:
        target_height = round(height * (target_width / width))

    ratio = max(target_width / width, target_height / height)
    new_width = max(1, round(width * ratio))
    new_height = max(1, round(height * ratio))

    resized = comfy.utils.common_upscale(
        images.movedim(-1, 1),
        new_width,
        new_height,
        upscale_method,
        crop="disabled",
    ).movedim(1, -1)

    left = max(0, (new_width - target_width) // 2)
    top = max(0, (new_height - target_height) // 2)
    right = left + target_width
    bottom = top + target_height

    resized = resized[:, top:bottom, left:right, :]

    return resized.clamp(0, 1)


def _get_latent_index(cond, latent_length, guide_length, frame_idx, scale_factors):
    time_scale_factor, _, _ = scale_factors
    # Negative frame_idx means positions before frame 0 (do not remap to end).
    if frame_idx >= 0 and guide_length > 1 and frame_idx != 0:
        frame_idx = (frame_idx - 1) // time_scale_factor * time_scale_factor + 1

    latent_idx = (frame_idx + time_scale_factor - 1) // time_scale_factor
    return frame_idx, latent_idx


def _adjust_negative_frame_idx(frame_idx, guide_length, time_scale_factor, negative_frame_mode):
    if negative_frame_mode != "before_start":
        return frame_idx

    if frame_idx < 0 and guide_length > 1:
        frame_idx = frame_idx - time_scale_factor * (guide_length - 1)
    return frame_idx


def _format_id_list(ids):
    if not ids:
        return "[]"
    if len(ids) <= 10:
        return "[" + ", ".join(str(i) for i in ids) + "]"
    return f"[{ids[0]}..{ids[-1]}] (len={len(ids)})"


def _build_info_text(
    frame_step,
    mask_mode,
    ramp_frames,
    negative_frame_mode,
    upscale_method,
    guide_info,
    step_multiplier=None,
):
    header_parts = [
        f"frame_step={frame_step}",
        f"mask_mode={mask_mode}",
        f"ramp_frames={ramp_frames}",
        f"negative_frame_mode={negative_frame_mode}",
        f"upscale_method={upscale_method}",
    ]
    if step_multiplier is not None:
        header_parts.insert(1, f"step_multiplier={step_multiplier}")
        header_parts.insert(2, f"effective_step={frame_step * step_multiplier}")

    lines = ["; ".join(header_parts)]
    lines.extend(guide_info)
    return "\n".join(lines)


class LTXVAddGuideMultiJsonFc:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "mask_mode": (
                    ["constant", "ramp"],
                    {"default": "constant"},
                ),
                "ramp_frames": (
                    "INT",
                    {"default": 1, "min": 1, "max": 64, "step": 1},
                ),
                "upscale_method": (
                    ["nearest-exact", "bilinear", "lanczos"],
                    {"default": "nearest-exact"},
                ),
                "negative_frame_mode": (
                    ["allow_cross_zero", "before_start"],
                    {"default": "allow_cross_zero"},
                ),
                "guides_json": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "processed_images", "info")
    FUNCTION = "add"
    CATEGORY = "LTX2"

    def add(
        self,
        positive,
        negative,
        vae,
        latent,
        mask_mode,
        ramp_frames,
        upscale_method,
        negative_frame_mode,
        guides_json,
    ):
        guides = _parse_guides_json(guides_json)

        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = _get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        time_scale_factor = scale_factors[0]
        _, width_scale_factor, height_scale_factor = scale_factors
        target_width = latent_width * width_scale_factor
        target_height = latent_height * height_scale_factor

        processed_images = []
        guide_info = []

        for guide in guides:
            image_path = _resolve_image_path(guide["image"])
            img = _load_image_tensor(image_path)
            img = _resize_for_preprocess(img, target_width, target_height, upscale_method)
            if guide.get("preprocess", True):
                img = nodes_lt.LTXVPreprocess().execute(
                    img,
                    guide.get("preprocess_crf", 33),
                )[0]
            processed_images.append(img)
            f_idx = guide["frame_idx"]
            strength = guide["strength"]

            image_1, t = _encode_guide(vae, latent_width, latent_height, img, scale_factors)
            image_1, t = _maybe_expand_guide(image_1, t, mask_mode, ramp_frames)
            f_idx = _adjust_negative_frame_idx(
                f_idx,
                len(image_1),
                time_scale_factor,
                negative_frame_mode,
            )
            frame_idx, latent_idx = _get_latent_index(
                positive,
                latent_length,
                len(image_1),
                f_idx,
                scale_factors,
            )

            guide_frames = t.shape[2]
            latent_ids = list(range(latent_idx, latent_idx + guide_frames))
            frame_ids = [frame_idx + time_scale_factor * k for k in range(guide_frames)]
            guide_info.append(
                "guide#{idx}: frame_idx={frame_idx}; latent_ids={latent_ids}; frame_ids={frame_ids}; strength={strength}".format(
                    idx=len(guide_info) + 1,
                    frame_idx=frame_idx,
                    latent_ids=_format_id_list(latent_ids),
                    frame_ids=_format_id_list(frame_ids),
                    strength=strength,
                )
            )

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
                mask_mode,
                ramp_frames,
            )

        processed_batch = torch.cat(processed_images, dim=0)
        info_text = _build_info_text(
            frame_step=time_scale_factor,
            mask_mode=mask_mode,
            ramp_frames=ramp_frames,
            negative_frame_mode=negative_frame_mode,
            upscale_method=upscale_method,
            guide_info=guide_info,
        )
        return (
            positive,
            negative,
            {"samples": latent_image, "noise_mask": noise_mask},
            processed_batch,
            info_text,
        )


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
                    io.Boolean.Input(
                        f"preprocess_{i}",
                        default=True,
                        tooltip=f"Apply LTXVPreprocess to guide {i} before encoding.",
                    ),
                    io.Int.Input(
                        f"preprocess_crf_{i}",
                        default=33,
                        min=0,
                        max=51,
                        step=1,
                        tooltip=f"CRF value for LTXVPreprocess for guide {i}.",
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
                io.Combo.Input(
                    "mask_mode",
                    options=["constant", "ramp"],
                    default="constant",
                    tooltip="Select how to build the guide noise mask.",
                ),
                io.Int.Input(
                    "ramp_frames",
                    default=1,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Expand single-frame guides to this many frames when using ramp.",
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=["nearest-exact", "bilinear", "lanczos"],
                    default="nearest-exact",
                    tooltip="Resize method used before preprocessing.",
                ),
                io.Combo.Input(
                    "negative_frame_mode",
                    options=["allow_cross_zero", "before_start"],
                    default="allow_cross_zero",
                    tooltip="How to place negative frame indices when guide spans multiple frames.",
                ),
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
                io.Image.Output(display_name="processed_images", tooltip="Batch of processed guide images"),
                io.String.Output(display_name="info", tooltip="Guide index mapping info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        latent,
        mask_mode,
        ramp_frames,
        upscale_method,
        negative_frame_mode,
        num_guides,
    ) -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = _get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        _, width_scale_factor, height_scale_factor = scale_factors
        target_width = latent_width * width_scale_factor
        target_height = latent_height * height_scale_factor

        image_keys = sorted([k for k in num_guides.keys() if k.startswith("image_")])

        processed_images = []
        guide_info = []

        for img_key in image_keys:
            i = img_key.split("_")[1]

            img = num_guides[f"image_{i}"]
            f_idx = num_guides[f"frame_idx_{i}"]
            strength = num_guides[f"strength_{i}"]
            img = _resize_for_preprocess(img, target_width, target_height, upscale_method)

            if num_guides.get(f"preprocess_{i}", True):
                img = nodes_lt.LTXVPreprocess().execute(
                    img,
                    num_guides.get(f"preprocess_crf_{i}", 33),
                )[0]
            processed_images.append(img)

            image_1, t = _encode_guide(vae, latent_width, latent_height, img, scale_factors)
            image_1, t = _maybe_expand_guide(image_1, t, mask_mode, ramp_frames)

            f_idx = _adjust_negative_frame_idx(
                f_idx,
                len(image_1),
                scale_factors[0],
                negative_frame_mode,
            )
            frame_idx, latent_idx = _get_latent_index(
                positive,
                latent_length,
                len(image_1),
                f_idx,
                scale_factors,
            )

            guide_frames = t.shape[2]
            latent_ids = list(range(latent_idx, latent_idx + guide_frames))
            frame_ids = [frame_idx + scale_factors[0] * k for k in range(guide_frames)]
            guide_info.append(
                "guide#{idx}: frame_idx={frame_idx}; latent_ids={latent_ids}; frame_ids={frame_ids}; strength={strength}".format(
                    idx=len(guide_info) + 1,
                    frame_idx=frame_idx,
                    latent_ids=_format_id_list(latent_ids),
                    frame_ids=_format_id_list(frame_ids),
                    strength=strength,
                )
            )
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
                mask_mode,
                ramp_frames,
            )

        processed_batch = torch.cat(processed_images, dim=0)
        info_text = _build_info_text(
            frame_step=scale_factors[0],
            mask_mode=mask_mode,
            ramp_frames=ramp_frames,
            negative_frame_mode=negative_frame_mode,
            upscale_method=upscale_method,
            guide_info=guide_info,
        )
        return io.NodeOutput(
            positive,
            negative,
            {"samples": latent_image, "noise_mask": noise_mask},
            processed_batch,
            info_text,
        )


class LTXVAddRefMultiFc(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = []
        for num_refs in range(1, 21):
            ref_inputs = []
            for i in range(1, num_refs + 1):
                ref_inputs.extend([
                    io.Image.Input(f"image_{i}"),
                    io.Float.Input(
                        f"strength_{i}",
                        default=1.0,
                        min=0.0,
                        max=1.0,
                        step=0.01,
                        tooltip=f"Strength for reference {i}.",
                    ),
                    io.Boolean.Input(
                        f"preprocess_{i}",
                        default=True,
                        tooltip=f"Apply LTXVPreprocess to reference {i} before encoding.",
                    ),
                    io.Int.Input(
                        f"preprocess_crf_{i}",
                        default=33,
                        min=0,
                        max=51,
                        step=1,
                        tooltip=f"CRF value for LTXVPreprocess for reference {i}.",
                    ),
                ])
            options.append(io.DynamicCombo.Option(
                key=str(num_refs),
                inputs=ref_inputs,
            ))

        return io.Schema(
            node_id="LTXVAddRefMultiFc",
            category="KJNodes/ltxv",
            description=(
                "Add multiple reference images at automatically assigned negative frame indices, "
                "uses DynamicCombo which requires ComfyUI 0.8.1 and frontend 1.33.4 or later."
            ),
            inputs=[
                io.Conditioning.Input("positive", tooltip="Positive conditioning to which guide keyframe info will be added"),
                io.Conditioning.Input("negative", tooltip="Negative conditioning to which guide keyframe info will be added"),
                io.Vae.Input("vae", tooltip="Video VAE used to encode the guide images"),
                io.Latent.Input("latent", tooltip="Video latent, guides are added to the end of this latent"),
                io.Int.Input(
                    "step_multiplier",
                    default=2,
                    min=1,
                    max=100,
                    step=1,
                    tooltip="Multiply frame_step to space reference frames.",
                ),
                io.Combo.Input(
                    "mask_mode",
                    options=["constant", "ramp"],
                    default="constant",
                    tooltip="Select how to build the guide noise mask.",
                ),
                io.Int.Input(
                    "ramp_frames",
                    default=1,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Expand single-frame refs to this many frames when using ramp.",
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=["nearest-exact", "bilinear", "lanczos"],
                    default="nearest-exact",
                    tooltip="Resize method used before preprocessing.",
                ),
                io.Combo.Input(
                    "negative_frame_mode",
                    options=["allow_cross_zero", "before_start"],
                    default="allow_cross_zero",
                    tooltip="How to place negative frame indices when refs span multiple frames.",
                ),
                io.DynamicCombo.Input(
                    "num_refs",
                    options=options,
                    display_name="Number of Refs",
                    tooltip="Select how many reference images to use",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent", tooltip="Video latent with added references"),
                io.Image.Output(display_name="processed_images", tooltip="Batch of processed reference images"),
                io.Int.Output(display_name="frame_step", tooltip="Frame step (vae.downscale_index_formula[0])"),
                io.String.Output(display_name="info", tooltip="Reference index mapping info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        latent,
        step_multiplier,
        mask_mode,
        ramp_frames,
        upscale_method,
        negative_frame_mode,
        num_refs,
    ) -> io.NodeOutput:
        scale_factors = vae.downscale_index_formula
        frame_step = scale_factors[0]
        effective_step = frame_step * step_multiplier
        latent_image = latent["samples"]
        noise_mask = _get_noise_mask(latent)

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        _, width_scale_factor, height_scale_factor = scale_factors
        target_width = latent_width * width_scale_factor
        target_height = latent_height * height_scale_factor

        image_keys = sorted([k for k in num_refs.keys() if k.startswith("image_")])
        processed_images = []
        guide_info = []

        for img_key in image_keys:
            i = img_key.split("_")[1]

            img = num_refs[f"image_{i}"]
            strength = num_refs[f"strength_{i}"]
            frame_idx = -effective_step * int(i)

            img = _resize_for_preprocess(img, target_width, target_height, upscale_method)
            if num_refs.get(f"preprocess_{i}", True):
                img = nodes_lt.LTXVPreprocess().execute(
                    img,
                    num_refs.get(f"preprocess_crf_{i}", 33),
                )[0]
            processed_images.append(img)

            image_1, t = _encode_guide(vae, latent_width, latent_height, img, scale_factors)
            image_1, t = _maybe_expand_guide(image_1, t, mask_mode, ramp_frames)
            frame_idx = _adjust_negative_frame_idx(
                frame_idx,
                len(image_1),
                scale_factors[0],
                negative_frame_mode,
            )
            frame_idx, latent_idx = _get_latent_index(
                positive,
                latent_length,
                len(image_1),
                frame_idx,
                scale_factors,
            )

            guide_frames = t.shape[2]
            latent_ids = list(range(latent_idx, latent_idx + guide_frames))
            frame_ids = [frame_idx + frame_step * k for k in range(guide_frames)]
            guide_info.append(
                "ref#{idx}: frame_idx={frame_idx}; latent_ids={latent_ids}; frame_ids={frame_ids}; strength={strength}".format(
                    idx=len(guide_info) + 1,
                    frame_idx=frame_idx,
                    latent_ids=_format_id_list(latent_ids),
                    frame_ids=_format_id_list(frame_ids),
                    strength=strength,
                )
            )
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
                mask_mode,
                ramp_frames,
            )

        processed_batch = torch.cat(processed_images, dim=0)
        info_text = _build_info_text(
            frame_step=frame_step,
            mask_mode=mask_mode,
            ramp_frames=ramp_frames,
            negative_frame_mode=negative_frame_mode,
            upscale_method=upscale_method,
            guide_info=guide_info,
            step_multiplier=step_multiplier,
        )
        return io.NodeOutput(
            positive,
            negative,
            {"samples": latent_image, "noise_mask": noise_mask},
            processed_batch,
            frame_step,
            info_text,
        )


class LTX2R2VBrowserLLM(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTX2R2VBrowserLLM",
            display_name="LTX2 R2V Browser LLM",
            category="LTX2",
            description="Calls an OpenAI-compatible endpoint from the browser frontend.",
            inputs=[
                io.String.Input(
                    "system_prompt",
                    default="",
                    multiline=True,
                    tooltip="System prompt passed to the LLM.",
                ),
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="User prompt passed to the LLM.",
                ),
                io.String.Input(
                    "api_endpoint",
                    default="https://api.openai.com/v1/chat/completions",
                    tooltip="OpenAI-compatible chat completions endpoint.",
                ),
                io.String.Input(
                    "model",
                    default="gpt-4o-mini",
                    tooltip="Model name to request.",
                ),
                io.String.Input(
                    "api_key",
                    default="",
                    tooltip="API key (stored in browser cookie by frontend).",
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional reference image input.",
                ),
                io.String.Input(
                    "response",
                    default="",
                    multiline=True,
                    tooltip="LLM response populated by the frontend.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="text"),
            ],
        )

    @classmethod
    def execute(
        cls,
        system_prompt,
        prompt,
        api_endpoint,
        model,
        api_key,
        response,
        image=None,
    ) -> io.NodeOutput:
        return io.NodeOutput(response or "")


NODE_CLASS_MAPPINGS = {
    "LTXVAddGuideMultiJsonFc": LTXVAddGuideMultiJsonFc,
    "LTXVAddGuideMultiFc": LTXVAddGuideMultiFc,
    "LTXVAddRefMultiFc": LTXVAddRefMultiFc,
    "LTX2R2VBrowserLLM": LTX2R2VBrowserLLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXVAddGuideMultiJsonFc": "LTXV Add Guide Multi (JSON) FC",
    "LTXVAddGuideMultiFc": "LTXV Add Guide Multi FC",
    "LTXVAddRefMultiFc": "LTXV Add Ref Multi FC",
    "LTX2R2VBrowserLLM": "LTX2 R2V Browser LLM",
}
