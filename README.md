# Microstructure Generation using DDPM (Diffusion Models)

## Project Overview
This project trains a **Denoising Diffusion Probabilistic Model (DDPM)** to generate realistic **microstructure images** (such as EBSD IPF maps). The deep generative model learns the statistical and crystallographic structure of micrographs and synthesizes new samples resembling real microstructures.

## Dataset Structure
The dataset is expected to follow this format:

```
dataset_root/
    Class1/
        Images/
            img1.png
            img2.png
    Class2/
        Images/
            img1.png
            img2.png
    ...
```

Each class may correspond to:
- deformation mode  
- strain path  
- alloy type  
- material state  

All images must be loaded as **RGB** because IPF maps encode orientation into color.

## What is a DDPM?

A DDPM consists of two processes:

### **1. Forward Diffusion (Noise Addition)**
The model progressively corrupts an image by adding Gaussian noise:

$$
x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1 - \alpha_t}\,\epsilon
$$

After ~1000 steps, the image becomes pure noise.

### **2. Reverse Diffusion (Denoising)**
A neural network (UNet) learns to reverse the noising process:

$$
\hat{\epsilon}_\theta(x_t, t) \approx \epsilon
$$

Sampling starts from random noise and iteratively denoises it to produce a realistic microstructure.

---

## UNet Architecture (for Microstructure Generation)

The UNet is composed of:

- **Downsampling path (Encoder)** — captures grain-level features  
- **Bottleneck with Attention** — captures long-range interactions between grains  
- **Upsampling path (Decoder)** — reconstructs the image, merging global + local features  
- **Skip connections** — preserve boundary information  
- **Class embeddings** (optional) — enable class-conditional generation  

The model takes:
- noisy image  
- timestep \( t \)  
- (optional) class label  

and predicts the added noise.

---

## Training Pipeline

### **1. Load and preprocess images**
- resized to 128×128  
- normalized to [-1, 1]  
- loaded as RGB  

### **2. Generate noisy images**
Random timestep `t` is sampled per image:

```python
noisy_imgs, noise, timesteps = get_noise_targets(imgs)
```

### **3. Forward pass**
```python
pred_noise = unet(noisy_imgs, timesteps, class_labels)
```

### **4. Loss**
The loss is:

$$
L = \lVert \hat{\epsilon}_\theta - \epsilon \rVert^2
$$

### **5. Optimization**
- AdamW optimizer  
- Mixed precision (`torch.autocast("cuda")`)  
- EMA (Exponential Moving Average) to stabilize sampling  

---

## Sampling (Generating Microstructures)

To generate samples:
1. Start from Gaussian noise
2. Iteratively denoise for 1000 → 0 steps
3. Apply optional class conditioning

Sampling produces an image resembling real IPF microstructures.

---

## What the Model Learns
The model implicitly captures:

- grain morphology  
- orientation gradients (similar to KAM)  
- texture clusters  
- crystallographic orientation distributions  
- grain-boundary patterns  
- phase/region distributions  

Because IPF colors encode orientation in RGB, the model learns crystallographic structure statistically.

---

## Files in This Repository

```
Class/                     # Dataset folder containing class-wise image directories
.python-version            # Python version lock (for environment reproducibility)
ddpm-microgen.ipynb        # Main Jupyter Notebook for training & sampling DDPM
main.py                    # Python script version of the DDPM pipeline (optional)
pyproject.toml             # Project + dependency metadata for Poetry/Pip
README.md                  # Project documentation
requirements.txt           # Python package dependencies for pip installations
```

---

## Future Improvements
- DDIM sampling for faster generation  
- Classifier-free guidance  
- Higher-resolution UNets  
- Conditional microstructure property generation  
- Grain-size–controlled synthesis  

---

## Acknowledgements
Thanks to:
- HuggingFace Diffusers  
- PyTorch Team  
- Materials science community  
