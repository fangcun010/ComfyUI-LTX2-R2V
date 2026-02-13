import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const EXTENSION_NAME = "LTX2-R2V.BrowserLLM";
const COOKIE_KEY = "ltx2r2v_api_key";

function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) {
    return parts.pop().split(";").shift();
  }
  return "";
}

function setCookie(name, value, days) {
  const expires = days
    ? `; expires=${new Date(Date.now() + days * 864e5).toUTCString()}`
    : "";
  document.cookie = `${name}=${value || ""}${expires}; path=/`;
}

function getWidget(node, name) {
  if (!node || !node.widgets) {
    return null;
  }
  return node.widgets.find((w) => w.name === name) || null;
}

function getInputImageBase64(node) {
  if (!node || !node.inputs) {
    return null;
  }
  const imageInput = node.inputs.find((i) => i && i.name === "image");
  if (!imageInput || !imageInput.link) {
    return null;
  }
  const link = node.graph.links[imageInput.link];
  if (!link) {
    return null;
  }
  const imageNode = node.graph.getNodeById(link.origin_id);
  if (!imageNode) {
    return null;
  }
  const imageWidget = imageNode.widgets?.find((w) => w.name === "image");
  if (!imageWidget || !imageWidget.value) {
    return null;
  }
  const fileName = imageWidget.value;
  if (!fileName) {
    return null;
  }
  return new Promise((resolve, reject) => {
    api
      .getImage(fileName)
      .then((blob) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const result = reader.result;
          if (typeof result === "string") {
            const base64 = result.split(",")[1] || "";
            resolve(base64);
          } else {
            resolve("");
          }
        };
        reader.onerror = () => reject(new Error("Failed to read image blob"));
        reader.readAsDataURL(blob);
      })
      .catch((err) => reject(err));
  });
}

async function callChatCompletions({
  endpoint,
  apiKey,
  model,
  systemPrompt,
  prompt,
  imageBase64,
}) {
  if (!endpoint) {
    throw new Error("Missing api_endpoint");
  }
  if (!apiKey) {
    throw new Error("Missing API key");
  }
  if (!model) {
    throw new Error("Missing model");
  }

  const messages = [];
  if (systemPrompt) {
    messages.push({ role: "system", content: systemPrompt });
  }

  if (imageBase64) {
    messages.push({
      role: "user",
      content: [
        { type: "text", text: prompt || "" },
        {
          type: "image_url",
          image_url: { url: `data:image/png;base64,${imageBase64}` },
        },
      ],
    });
  } else {
    messages.push({ role: "user", content: prompt || "" });
  }

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model,
      messages,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  const text =
    data?.choices?.[0]?.message?.content ||
    data?.choices?.[0]?.text ||
    "";
  return text;
}

function ensureWidgets(node) {
  const responseWidget = getWidget(node, "response");
  if (responseWidget && responseWidget.type !== "text") {
    responseWidget.type = "text";
  }

  const apiKeyWidget = getWidget(node, "api_key");
  if (apiKeyWidget) {
    const stored = getCookie(COOKIE_KEY);
    if (stored && !apiKeyWidget.value) {
      apiKeyWidget.value = stored;
    }
  }
}

function addInvokeButton(node) {
  if (!node || !node.addWidget) {
    return;
  }
  const existing = node.widgets?.find((w) => w.name === "invoke");
  if (existing) {
    return;
  }
  node.addWidget("button", "invoke", "Invoke", async () => {
    const responseWidget = getWidget(node, "response");
    const apiEndpointWidget = getWidget(node, "api_endpoint");
    const apiKeyWidget = getWidget(node, "api_key");
    const modelWidget = getWidget(node, "model");
    const systemWidget = getWidget(node, "system_prompt");
    const promptWidget = getWidget(node, "prompt");

    if (!responseWidget) {
      return;
    }

    responseWidget.value = "Calling...";
    node.setDirtyCanvas(true, true);

    try {
      const apiKey = apiKeyWidget?.value || "";
      if (apiKeyWidget && apiKey) {
        setCookie(COOKIE_KEY, apiKey, 30);
      }

      const imageBase64 = await getInputImageBase64(node);

      const text = await callChatCompletions({
        endpoint: apiEndpointWidget?.value || "",
        apiKey,
        model: modelWidget?.value || "",
        systemPrompt: systemWidget?.value || "",
        prompt: promptWidget?.value || "",
        imageBase64,
      });

      responseWidget.value = text || "";
    } catch (err) {
      responseWidget.value = `Error: ${err?.message || err}`;
    }

    node.setDirtyCanvas(true, true);
  });
}

app.registerExtension({
  name: EXTENSION_NAME,
  async nodeCreated(node) {
    if (node?.comfyClass !== "LTX2R2VBrowserLLM") {
      return;
    }
    ensureWidgets(node);
    addInvokeButton(node);
  },
});
