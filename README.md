# GAN vs. DCGAN

A comparison of a standard fully connected GAN and a Deep Convolutional GAN (DCGAN) trained on the FashionMNIST dataset using PyTorch.

---

## Overview

This project implements two generative adversarial network architectures side by side to illustrate the structural and practical differences between a basic MLP-based GAN and the DCGAN architecture introduced by Radford et al. (2015). Both models are trained under identical hyperparameters to isolate the effect of architecture on image generation quality.

---

## Dataset

**FashionMNIST** — 70,000 grayscale images of clothing items across 10 categories (60,000 training / 10,000 test), each 28×28 pixels.

- Images are normalized to the range `[-1, 1]` to match the `Tanh` output activation used in both generators.
- Only the training split is used during GAN training (labels are ignored).
- The dataset is downloaded automatically via `torchvision` if not already present in `./data`.

---

## Why Compare GAN and DCGAN?

The original GAN (Goodfellow et al., 2014) uses fully connected layers throughout, treating images as flat vectors. This works but does not exploit the spatial structure of images. DCGAN replaces fully connected layers with convolutional and transposed convolutional layers, adds Batch Normalization, and uses architectural guidelines specifically designed for stable adversarial training on images. Comparing the two on the same dataset makes the practical impact of these design choices visible.

---

## Architecture Summary

### GAN — `6-2_GAN.ipynb`

**Generator (MLP)**

| Layer | Output Shape |
|---|---|
| `Linear(100, 256)` + `LeakyReLU(0.2)` | `(256,)` |
| `Linear(256, 512)` + `LeakyReLU(0.2)` | `(512,)` |
| `Linear(512, 1024)` + `LeakyReLU(0.2)` | `(1024,)` |
| `Linear(1024, 784)` + `Tanh` → reshape | `(1, 28, 28)` |

**Discriminator (MLP)**

| Layer | Output Shape |
|---|---|
| Flatten input | `(784,)` |
| `Linear(784, 512)` + `LeakyReLU(0.2)` | `(512,)` |
| `Linear(512, 256)` + `LeakyReLU(0.2)` | `(256,)` |
| `Linear(256, 1)` (raw logit) | `(1,)` |

---

### DCGAN — `6-3_DCGAN.ipynb`

**Generator (CNN)**

| Layer | Output Shape |
|---|---|
| Input noise reshaped to `(100, 1, 1)` | `(100, 1, 1)` |
| `ConvTranspose2d(100→128, k=7, s=1)` + `BN` + `ReLU` | `(128, 7, 7)` |
| `ConvTranspose2d(128→64, k=4, s=2, p=1)` + `BN` + `ReLU` | `(64, 14, 14)` |
| `ConvTranspose2d(64→1, k=4, s=2, p=1)` + `Tanh` | `(1, 28, 28)` |

**Discriminator (CNN)**

| Layer | Output Shape |
|---|---|
| `Conv2d(1→64, k=4, s=2, p=1)` + `LeakyReLU(0.2)` | `(64, 14, 14)` |
| `Conv2d(64→128, k=4, s=2, p=1)` + `BN` + `LeakyReLU(0.2)` | `(128, 7, 7)` |
| `Conv2d(128→1, k=7, s=1)` (raw logit) | `(1, 1, 1)` → `(1,)` |

---

## Training Configuration

| Setting | Value |
|---|---|
| Dataset | FashionMNIST (training split) |
| Batch size | 128 |
| Latent dimension (`z_dim`) | 100 |
| Epochs | 30 |
| Optimizer | Adam, `lr=0.0002`, `betas=(0.5, 0.999)` |
| Loss function | `BCEWithLogitsLoss` |
| Weight init | Normal distribution, `mean=0`, `std=0.02` |

Both notebooks use identical hyperparameters.

---

## Key Architectural Differences

### Fully Connected vs. Convolutional

The GAN generator maps the latent vector through a series of `Linear` layers and reshapes the flat output into an image. The DCGAN generator reshapes the latent vector into a small spatial feature map first and progressively upsamples it using `ConvTranspose2d`. This allows the network to learn spatially local features and preserves the 2D structure of images throughout generation.

### ConvTranspose2d (Transposed Convolution)

`ConvTranspose2d` performs a learnable fractionally-strided convolution that increases spatial resolution. In the DCGAN generator, three successive transposed convolutions upsample from `(100, 1, 1)` → `(128, 7, 7)` → `(64, 14, 14)` → `(1, 28, 28)`. This is qualitatively different from interpolation-based upsampling: the kernel weights are learned and the upsampling is part of the optimization.

### Batch Normalization in DCGAN

`BatchNorm2d` is applied after each `ConvTranspose2d` in the generator (except the output layer) and after the second `Conv2d` in the discriminator. Batch Normalization normalizes the activations within each mini-batch, which stabilizes gradient flow, reduces sensitivity to weight initialization, and helps both networks train more reliably. The basic GAN does not use Batch Normalization.

### LeakyReLU in Discriminators

Both discriminators use `LeakyReLU(0.2)` rather than standard `ReLU`. A standard ReLU outputs zero for all negative activations, which can cause neurons to permanently stop contributing to the gradient (the "dying ReLU" problem). `LeakyReLU` passes a small fraction (0.2) of negative activations through, keeping gradients alive and helping the discriminator continue learning throughout training.

### BCEWithLogitsLoss vs BCELoss

Both notebooks use `BCEWithLogitsLoss`, which combines a sigmoid activation and binary cross-entropy into a single numerically stable operation. The discriminators output raw logits (no final sigmoid), and the loss function handles the sigmoid internally using the log-sum-exp trick. This avoids the floating point instability that can occur when computing `log(sigmoid(x))` separately for very large or very small values of `x`.

---

## Visualization: fixed_noise

Both notebooks define a `fixed_noise` tensor (25 samples) **outside** the training loop and reuse it at the end of every epoch to generate a 5×5 grid of images for display. Because the noise is fixed, the same 25 latent points are decoded at every epoch, making it possible to track how individual generated samples evolve over training. If new random noise were sampled each epoch, apparent changes in the images could not be attributed to genuine learning.

---

## Reproducibility Note

Neither notebook sets a global random seed (`torch.manual_seed`, `np.random.seed`). Generator and discriminator weights are initialized stochastically, and training data is shuffled each epoch. As a result, generated image quality and loss curves will vary between runs. This is expected behavior for GAN training and does not indicate an error.

---

## Setup and Usage

**Requirements**

Install dependencies (see `requirements.txt`):

```bash
pip install -r requirements.txt
```

**Running the notebooks**

```bash
jupyter notebook
```

Open `6-2_GAN.ipynb` or `6-3_DCGAN.ipynb` and run all cells in order.

- The dataset downloads automatically on first run if not already in `./data`.
- Training runs for 30 epochs. A 5×5 grid of generated images is displayed after each epoch.
- Loss curves for generator and discriminator are plotted at the end.
- GPU is used automatically if available (`cuda`); otherwise CPU is used.

**Expected output**

During training, progress is printed every 100 batches:
```
[Epoch 1/30] [Batch 0/469] [D loss: 0.6931] [G loss: 0.6931]
```

After each epoch, a 5×5 grid of generated images is displayed inline. Early epochs will produce noisy or blurry outputs. By epoch 20–30, the DCGAN should produce visibly sharper and more structured clothing shapes than the basic GAN.

---

## Files

| File | Description |
|---|---|
| `6-2_GAN.ipynb` | Fully connected GAN implementation |
| `6-3_DCGAN.ipynb` | Deep Convolutional GAN implementation |
| `data/FashionMNIST/` | FashionMNIST dataset (auto-downloaded) |
| `requirements.txt` | Python dependencies |

---

## Reference

Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* arXiv:1511.06434. https://arxiv.org/abs/1511.06434
