# Question

Explain the output (shapes) of the activations during forward pass and gradients during backward:

```text
conv1: activations torch.Size([1, 64, 112, 112])
layer1.0: activations torch.Size([1, 256, 56, 56])
layer4.0: activations torch.Size([1, 2048, 7, 7])
layer4.2: activations torch.Size([1, 2048, 7, 7])

layer4.2: gradients torch.Size([1, 2048, 7, 7])
layer4.0: gradients torch.Size([1, 2048, 7, 7])
layer1.0: gradients torch.Size([1, 256, 56, 56])
conv1: gradients torch.Size([1, 64, 112, 112])
```

# Answer

Those shapes are exactly what you should expect in ResNet-50.

`[1, C, H, W]` means:

- `1`: batch size, one image
- `C`: number of channels / feature maps
- `H, W`: spatial resolution of each feature map

Forward activations:

- `conv1: [1, 64, 112, 112]`
  The first conv produces 64 low-level feature maps at fairly high spatial resolution. This stage still preserves a lot of local detail like edges, corners, and texture.

- `layer1.0: [1, 256, 56, 56]`
  After early downsampling, spatial size is halved from `112x112` to `56x56`, while channels increase to 256. The network is trading spatial detail for richer feature representation.

- `layer4.0: [1, 2048, 7, 7]`
  Much deeper in the network, the feature map is very coarse spatially, only `7x7`, but extremely rich in channels. Each spatial cell now corresponds to a large receptive field in the input image.

- `layer4.2: [1, 2048, 7, 7]`
  Same shape as `layer4.0` because it is in the same ResNet stage. The difference is not tensor size, but content: `layer4.2` is later and usually more class-specific.

Backward gradients:

- `layer4.2: [1, 2048, 7, 7]`
- `layer4.0: [1, 2048, 7, 7]`
- `layer1.0: [1, 256, 56, 56]`
- `conv1: [1, 64, 112, 112]`

These match the activation shapes because each gradient is:

`d(target_score) / d(activation_tensor)`

So the gradient tensor must have the same shape as the activation tensor it is differentiating with respect to.

What the gradients mean:

- Each entry says how much changing that activation value would change the chosen class score.
- Positive large values mean "increasing this activation would help the target class."
- Small or negative values mean that location/channel is less helpful or suppressive.

Why Grad-CAM uses them:

- You average the gradients over `H, W` to get one weight per channel:
  `channel_weights = gradients.mean(dim=(2, 3), keepdim=True)`
- Then those channel weights tell you which activation maps mattered most for the class.
- Summing the weighted activations gives a 2D class-localization map.

Why early layers look detailed and late layers look coarse:

- `conv1` has `112x112`, so it can localize fine edges and contours.
- `layer4.*` has only `7x7`, so it loses precise detail but gains semantic meaning.
- After you upsample `7x7` to the input image size, it will look much blurrier but often more class-focused.

Why backward order looks reversed:

- Forward goes `conv1 -> ... -> layer4.2`
- Backward goes from the target score back through the network, so you see:
  `layer4.2 -> layer4.0 -> layer1.0 -> conv1`

That reverse order is normal backpropagation.

Short interpretation:

- Activations tell you what the layer detected.
- Gradients tell you which of those detections mattered for the chosen class.
- Same shape is expected because the gradients are attached to those exact activation tensors.
