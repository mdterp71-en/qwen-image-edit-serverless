"""
RunPod Serverless Handler for Qwen-Image-Edit-2509-LoRAs-Fast-Lazy-Load

Wraps the Lazy-Load Gradio app's pipeline as a serverless API.
Accepts a base64 image + prompt + adapter name, returns edited image as base64.

All 8 LoRA adapters from the Lazy-Load repo are supported and loaded on-demand.
"""

import runpod
import torch
import base64
import random
import os
from io import BytesIO
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
pipe = None
LOADED_LORA = None  # tracks which adapter is currently active

# ---------------------------------------------------------------------------
# Adapter registry — matches the Lazy-Load repo's ADAPTER_SPECS exactly
# (repo_id, filename)
# ---------------------------------------------------------------------------
ADAPTER_MAP = {
    "Photo-to-Anime": (
        "autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime",
        "Qwen-Image-Edit-2509-Photo-to-Anime_000001000.safetensors",
    ),
    "Multiple-Angles": (
        "dx8152/Qwen-Edit-2509-Multiple-angles",
        "镜头转换.safetensors",
    ),
    "Light-Restoration": (
        "dx8152/Qwen-Image-Edit-2509-Light_restoration",
        "移除光影.safetensors",
    ),
    "Relight": (
        "dx8152/Qwen-Image-Edit-2509-Relight",
        "Qwen-Edit-Relight.safetensors",
    ),
    "Multi-Angle-Lighting": (
        "flymy-ai/qwen-image-edit-2509-multi-angle-lighting-lora",
        "pytorch_lora_weights.safetensors",
    ),
    "Edit-Skin": (
        "flymy-ai/qwen-image-edit-2509-edit-skin-lora",
        "pytorch_lora_weights.safetensors",
    ),
    "Next-Scene": (
        "lovis93/next-scene-qwen-image-lora-2509",
        "next-scene_lora_v1-3000.safetensors",
    ),
    "Upscale-Image": (
        "jasperai/qwen-image-edit-2509-upscaler-lora",
        "pytorch_lora_weights.safetensors",
    ),
}

MAX_SIZE = 1024
NEGATIVE_PROMPT = (
    "lowres, watermark, banner, logo, contactinfo, text, deformed, blurry, "
    "blur, out of focus, out of frame, surreal, extra, ugly, upholstered, "
    "fabric, grainy, distorted, disfigured, poorly drawn, bad anatomy, "
    "wrong anatomy, extra limb, missing limb, floating limbs, extra fingers, "
    "extra digit, fewer digits"
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_pipeline():
    """Load base pipeline + fast Rapid-AIO transformer."""
    global pipe

    # Try importing the custom pipeline class from the repo's qwenimage/ folder
    # Falls back to standard QwenImageEditPipeline from diffusers
    try:
        from qwenimage.pipeline_qwen_image_edit import QwenImageEditPipeline
        print("Using custom pipeline from qwenimage/")
    except ImportError:
        from diffusers import QwenImageEditPipeline
        print("Using standard diffusers QwenImageEditPipeline")

    print("Loading Qwen-Image-Edit-2509 base pipeline...")
    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        torch_dtype=torch.bfloat16,
    )

    # Swap in the Rapid-AIO transformer for 4-step fast inference
    print("Loading Rapid-AIO transformer...")
    pipe.transformer = pipe.transformer.from_pretrained(
        "linoyts/Qwen-Image-Edit-Rapid-AIO",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    pipe.scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=8.0,
    )

    # Try enabling Flash Attention 3 (optional optimization)
    try:
        from qwenimage.attention_processor import QwenDoubleStreamAttnProcessorFA3
        pipe.transformer.set_attn_processor(
            QwenDoubleStreamAttnProcessorFA3()
        )
        print("Flash Attention 3 enabled")
    except Exception as e:
        print(f"FA3 not available, using default attention: {e}")

    pipe.to("cuda")
    print("Pipeline loaded and ready on CUDA")


def apply_lora(adapter_name: str):
    """Hot-swap LoRA adapter. Skips if already loaded."""
    global LOADED_LORA

    if adapter_name == LOADED_LORA:
        return

    if LOADED_LORA is not None:
        pipe.unload_lora_weights()

    repo_id, filename = ADAPTER_MAP[adapter_name]
    lora_path = hf_hub_download(repo_id=repo_id, filename=filename)
    pipe.load_lora_weights(lora_path)
    LOADED_LORA = adapter_name
    print(f"LoRA loaded: {adapter_name}")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def resize_image(img: Image.Image, max_size: int = MAX_SIZE) -> Image.Image:
    """Resize keeping aspect ratio, dimensions rounded to multiples of 8."""
    w, h = img.size
    scale = min(max_size / w, max_size / h, 1.0)
    new_w = int(w * scale) // 8 * 8
    new_h = int(h * scale) // 8 * 8
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_image(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(event):
    """
    Expected input:
    {
        "input": {
            "image":          "<base64 encoded image>",
            "prompt":         "Transform into anime style",
            "adapter":        "Photo-to-Anime",       (optional, default: Photo-to-Anime)
            "seed":           -1,                      (optional, -1 = random)
            "guidance_scale": 3.5,                     (optional)
            "num_steps":      4                        (optional)
        }
    }

    Returns:
    {
        "image":   "<base64 PNG>",
        "seed":    int,
        "adapter": str,
        "width":   int,
        "height":  int
    }
    """
    global pipe

    if pipe is None:
        load_pipeline()

    inp = event["input"]

    # --- Validate ---
    if "image" not in inp:
        return {"error": "Missing required field: 'image' (base64 encoded)"}
    if "prompt" not in inp:
        return {"error": "Missing required field: 'prompt'"}

    adapter = inp.get("adapter", "Photo-to-Anime")
    if adapter not in ADAPTER_MAP:
        return {
            "error": f"Unknown adapter '{adapter}'. "
                     f"Valid options: {list(ADAPTER_MAP.keys())}"
        }

    seed = int(inp.get("seed", -1))
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    guidance = float(inp.get("guidance_scale", 3.5))
    steps = int(inp.get("num_steps", 4))

    try:
        # Decode + resize input image
        src = base64_to_image(inp["image"])
        src = resize_image(src)
        w, h = src.size

        # Apply the requested LoRA
        apply_lora(adapter)

        # Run inference
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = pipe(
            image=src,
            prompt=inp["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            guidance_scale=guidance,
            num_inference_steps=steps,
            height=h,
            width=w,
            generator=generator,
        ).images[0]

        return {
            "image": image_to_base64(result),
            "seed": seed,
            "adapter": adapter,
            "width": w,
            "height": h,
        }

    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
runpod.serverless.start({"handler": handler})
