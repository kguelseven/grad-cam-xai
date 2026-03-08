#!/usr/bin/env python3
"""Generate Grad-CAM visualizations for selected layers on a single image."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.error import URLError
from typing import Callable

import matplotlib

# Use a non-interactive backend so the script works in terminals and CI-like runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models import (
    GoogLeNet_Weights,
    ResNet50_Weights,
    googlenet,
    resnet50,
)


ModelFactory = Callable[..., nn.Module]


MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "resnet50": {
        "factory": resnet50,
        "weights_enum": ResNet50_Weights,
        "weights_name": "IMAGENET1K_V2",
        "description": "Recommended default for learning Grad-CAM on a linear CNN.",
    },
    "googlenet": {
        "factory": googlenet,
        "weights_enum": GoogLeNet_Weights,
        "weights_name": "IMAGENET1K_V1",
        "description": "Optional comparison model with inception branches.",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_TORCH_HOME = PROJECT_ROOT / ".torch-cache"


class GradCAMRecorder:
    """Capture activations and gradients for a target module."""

    def __init__(self, module: nn.Module, layer_name: str) -> None:
        self.layer_name = layer_name
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = module.register_forward_hook(self._forward_hook)

    def _forward_hook(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        del module, inputs
        self.activations = output.detach()
        print(f"{self.layer_name}: activations {self.activations.shape}")
        # hook after computing of the gradient
        output.register_hook(self._tensor_backward_hook)

    def _tensor_backward_hook(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()
        print(f"{self.layer_name}: gradients {self.gradients.shape}")

    def remove(self) -> None:
        self._forward_handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM overlays for one or more exact PyTorch module names "
            "using ImageNet-pretrained torchvision models."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Path to the input image. Required unless --list-layers is used.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS.keys()),
        default="resnet50",
        help="Pretrained model to use.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        help="Exact module names to visualize, e.g. layer4.2.conv3 or inception5b.branch4.1.conv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where overlay images and the comparison grid will be saved.",
    )
    parser.add_argument(
        "--class-idx",
        type=int,
        help="Optional target class index. Defaults to the model's top prediction.",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print available module names for the selected model and exit.",
    )
    return parser.parse_args()


def get_model_components(model_name: str, with_weights: bool) -> tuple[nn.Module, object]:
    config = MODEL_CONFIGS[model_name]
    factory = config["factory"]
    weights_enum = config["weights_enum"]
    weights_name = config["weights_name"]
    if not isinstance(factory, Callable):
        raise TypeError(f"Invalid model factory for {model_name}.")
    if not hasattr(weights_enum, weights_name):
        raise AttributeError(f"Invalid weights enum value for {model_name}: {weights_name}")
    weights = getattr(weights_enum, weights_name) if with_weights else None
    # Torchvision caches pretrained weights under TORCH_HOME. Pointing that cache at the
    # project keeps the example self-contained and avoids permission issues in locked-down
    # environments where ~/.cache is not writable.
    os.environ.setdefault("TORCH_HOME", str(LOCAL_TORCH_HOME))
    LOCAL_TORCH_HOME.mkdir(parents=True, exist_ok=True)
    try:
        model = factory(weights=weights)
    except (OSError, URLError) as error:
        raise SystemExit(
            "Failed to load pretrained weights. Run the script once with internet access "
            f"so torchvision can cache them under {LOCAL_TORCH_HOME}, then retry.\n"
            f"Original error: {error}"
        ) from error
    model.eval()
    return model, weights


def list_layers(model_name: str) -> int:
    model, _ = get_model_components(model_name, with_weights=False)
    print(f"Available layers for model={model_name}")
    for name, module in model.named_modules():
        if not name:
            continue
        print(f"{name}: {module.__class__.__name__}")
    return 0


def validate_args(args: argparse.Namespace) -> None:
    if args.list_layers:
        return
    if args.image is None:
        raise SystemExit("--image is required unless --list-layers is used.")
    if not args.image.is_file():
        raise SystemExit(f"Image file not found: {args.image}")
    if not args.layers:
        raise SystemExit("--layers is required unless --list-layers is used.")


def load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def resolve_layers(model: nn.Module, layer_names: list[str]) -> dict[str, nn.Module]:
    available_modules = dict(model.named_modules())
    resolved: dict[str, nn.Module] = {}
    invalid_names: list[str] = []
    for layer_name in layer_names:
        module = available_modules.get(layer_name)
        if module is None:
            invalid_names.append(layer_name)
            continue
        resolved[layer_name] = module
    if invalid_names:
        joined = ", ".join(invalid_names)
        raise SystemExit(
            f"Unknown layer name(s): {joined}\n"
            "Use --list-layers to inspect valid module names for the selected model."
        )
    return resolved


def ensure_spatial_layer(layer_name: str, activation: torch.Tensor) -> None:
    # Grad-CAM relies on spatial feature maps. Rejecting non-spatial tensors avoids
    # producing misleading heatmaps from classifier heads or flattened activations.
    if activation.ndim != 4:
        raise SystemExit(
            f"Layer '{layer_name}' produced shape {tuple(activation.shape)}. "
            "Grad-CAM requires a 4D activation map [N, C, H, W]."
        )


def preprocess_image(image: Image.Image, weights: object) -> tuple[torch.Tensor, np.ndarray]:
    # The pretrained weights define the exact resize, crop, and normalization pipeline.
    # Reusing that transform keeps the input distribution aligned with what the model saw
    # during ImageNet training, which makes the prediction and the Grad-CAM more reliable.
    preprocess = weights.transforms()
    input_tensor = preprocess(image).unsqueeze(0)
    image_array = np.asarray(image).astype(np.float32) / 255.0
    return input_tensor, image_array


def compute_gradcams(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_modules: dict[str, nn.Module],
    class_idx: int | None,
    output_size: tuple[int, int],
) -> tuple[dict[str, np.ndarray], int, torch.Tensor]:
    recorders = {name: GradCAMRecorder(module, name) for name, module in layer_modules.items()}
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]

        if class_idx is None:
            # default to top prediction
            class_idx = int(logits.argmax(dim=1).item())

        print(f"target_score = {logits[:, class_idx]}")
        target_score = logits[:, class_idx].sum()
        print(f"target_score = {target_score}")
        target_score.backward()

        cams: dict[str, np.ndarray] = {}
        for layer_name, recorder in recorders.items():
            if recorder.activations is None or recorder.gradients is None:
                raise RuntimeError(f"Hooks did not capture data for layer '{layer_name}'.")
            ensure_spatial_layer(layer_name, recorder.activations)

            # Grad-CAM uses the mean gradient per channel as an importance weight for the
            # corresponding activation map. This answers "which channels mattered most for
            # the chosen class score?" before collapsing them into a 2D heatmap.
            channel_weights = recorder.gradients.mean(dim=(2, 3), keepdim=True)
            weighted_activations = channel_weights * recorder.activations
            cam = weighted_activations.sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            cam = F.interpolate(
                cam,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            cam = cam.squeeze().cpu().numpy()
            cam -= cam.min()
            max_value = cam.max()
            if max_value > 0:
                cam /= max_value
            cams[layer_name] = cam
    finally:
        for recorder in recorders.values():
            recorder.remove()

    return cams, class_idx, logits.detach()


def overlay_heatmap(image_array: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color_map = matplotlib.colormaps["jet"]
    heatmap = color_map(cam)[..., :3]
    blended = (1 - alpha) * image_array + alpha * heatmap
    return np.clip(blended, 0.0, 1.0)


def save_outputs(
    image_array: np.ndarray,
    overlays: dict[str, np.ndarray],
    output_dir: Path,
    model_name: str,
    predicted_label: str,
    predicted_class_idx: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual overlays so you can inspect each layer on its own, then build a
    # grid to compare how attention shifts from shallow textures to deeper semantics.
    for layer_name, overlay in overlays.items():
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        output_path = output_dir / f"{model_name}_{safe_name}_overlay.png"
        plt.imsave(output_path, overlay)

    num_panels = len(overlays) + 1
    columns = min(3, num_panels)
    rows = int(np.ceil(num_panels / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows))
    axes_array = np.atleast_1d(axes).flatten()

    axes_array[0].imshow(image_array)
    axes_array[0].set_title(f"Original\n{predicted_label} (class {predicted_class_idx})")
    axes_array[0].axis("off")

    for axis, (layer_name, overlay) in zip(axes_array[1:], overlays.items(), strict=False):
        axis.imshow(overlay)
        axis.set_title(layer_name)
        axis.axis("off")

    for axis in axes_array[num_panels:]:
        axis.axis("off")

    figure.tight_layout()
    grid_path = output_dir / f"{model_name}_gradcam_grid.png"
    figure.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def format_top_predictions(logits: torch.Tensor, categories: list[str], limit: int = 5) -> str:
    probabilities = torch.softmax(logits, dim=1)
    values, indices = torch.topk(probabilities, k=limit, dim=1)
    lines: list[str] = []
    for rank, (score, idx) in enumerate(zip(values[0], indices[0], strict=False), start=1):
        class_idx = int(idx.item())
        class_name = categories[class_idx]
        lines.append(f"{rank}. {class_name} ({class_idx}) - {score.item():.4f}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    if args.list_layers:
        return list_layers(args.model)

    validate_args(args)
    model, weights = get_model_components(args.model, with_weights=True)
    if weights is None:
        raise RuntimeError(f"Expected pretrained weights for model '{args.model}'.")

    image = load_image(args.image)
    input_tensor, image_array = preprocess_image(image, weights)
    layer_modules = resolve_layers(model, args.layers)
    cams, target_class_idx, logits = compute_gradcams(
        model=model,
        input_tensor=input_tensor,
        layer_modules=layer_modules,
        class_idx=args.class_idx,
        output_size=image_array.shape[:2],
    )

    categories = list(weights.meta["categories"])
    predicted_label = categories[target_class_idx]
    overlays = {
        layer_name: overlay_heatmap(image_array, cam) for layer_name, cam in cams.items()
    }
    save_outputs(
        image_array=image_array,
        overlays=overlays,
        output_dir=args.output_dir,
        model_name=args.model,
        predicted_label=predicted_label,
        predicted_class_idx=target_class_idx,
    )

    print(f"Model: {args.model}")
    print(f"Target class: {predicted_label} ({target_class_idx})")
    print("Top predictions:")
    print(format_top_predictions(logits, categories))
    print(f"Saved outputs to: {args.output_dir.resolve()}")
    for layer_name in overlays:
        print(f" - {layer_name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
