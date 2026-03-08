# Implementation Plan

## Summary

Build a single Python script that runs Grad-CAM on a user-provided dog image, lets you choose exact target layers by PyTorch module name, and saves both a comparison grid and per-layer overlay images.

Default to `ResNet50_Weights.IMAGENET1K_V2`, with `GoogLeNet_Weights.IMAGENET1K_V1` supported as an optional model choice.

## Key Decisions

- Use `resnet50` as the teaching default because Grad-CAM is easier to interpret on a linear CNN.
- Keep `googlenet` as a supported alternative for comparison.
- Require exact layer names through `--layers`.
- Save artifacts to disk by default instead of opening a GUI window.
- Use the existing project `.venv` as the execution target.
- Add concise comments in the relevant code sections that explain both what the code does and why it matters.

## Deliverables

- `grad_cam_demo.py` as the runnable example
- `README.md` with setup and usage
- per-layer Grad-CAM overlays and a combined grid written to `outputs/`

## Acceptance Criteria

- User can inspect module names with `--list-layers`
- User can pass a local dog image and selected layers
- Script generates one heatmap overlay per selected layer
- Script generates a comparison grid
- Invalid or non-spatial layers fail clearly
