# project-vision: Continuous Event-Stream Frame Reconstruction via Implicit Neural Representations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
<!-- [UPDATE LATER: Add CI/CD, paper preprint arXiv badge, or pretrained model weights badge once available] -->

An algorithmic deep learning framework to reconstruct continuous, ultra-sharp video frames from asynchronous neuromorphic event streams using continuous coordinate networks (Implicit Neural Representations).

---

## 📌 Project Overview

Standard cameras capture visual scenes at fixed exposure intervals, leading to severe **motion blur** during high-speed movement and **over/under-exposure** in extreme lighting conditions. 

Neuromorphic event cameras (such as the DAVIS or Prophesee sensors) eliminate standard frames entirely. Instead, each pixel reacts asynchronously to microsecond brightness changes, generating continuous tuples:
$$(x, y, t, p)$$
where $x, y$ are spatial pixel coordinates, $t$ is microsecond-accurate time, and $p \in \{-1, +1\}$ is polarity.

`project-vision` formulates a continuous neural coordinate function:
$$f_\theta(x, y, t) \rightarrow I$$
mapping spatio-temporal coordinates directly to intensity values. By learning this continuous representation, we achieve:
1. **Zero temporal quantisation:** Query frames at arbitrary temporal resolution (e.g., 100 fps, 10,000 fps) without fixed voxel binning.
2. **Drift-free reconstruction:** Preserving contrast and edges during static/slow camera intervals where events become sparse.
3. **Purely software-driven:** Trained and evaluated using open public benchmarks (DAVIS240C, MVSEC, HQF).

---

## 📁 Repository Structure

```text
project-vision/
├── data/                   # Dataset symlinks and cached event tensors (git-ignored)
│   └── raw/                # Raw .aedat / .h5 downloads (DAVIS240C, MVSEC)
├── configs/                # YAML configuration files for models and training
│   ├── base_inr.yaml       # Default coordinate MLP architecture config
│   └── exp_baseline.yaml   # Baseline comparison settings
├── src/
│   ├── data/               # Event data loading and parsing via Tonic
│   │   ├── dataset.py      # Custom PyTorch Dataset wrapping event streams
│   │   └── transforms.py   # Spatio-temporal coordinate normalisation
│   ├── models/             # Neural network definitions
│   │   ├── inr.py          # Continuous coordinate MLP with Fourier features
│   │   └── layers.py       # Positional encodings & continuous activation blocks
│   ├── losses/             # Loss formulation (L1, differential event loss, LPIPS)
│   │   └── event_loss.py
│   └── utils/              # Metrics (PSNR, SSIM, LPIPS), logging, and visualization
│       ├── metrics.py
│       └── render.py       # Continuous frame query and video rendering tools
├── notebooks/              # Exploration and visual debugging notebooks
│   └── 01_explore_events.ipynb
├── train.py                # Main training execution script
├── evaluate.py             # Evaluation and benchmark metrics generation
├── environment.yml         # Conda environment specifications
├── requirements.txt        # PIP dependencies
└── README.md
```

---

## ⚙️ Environment Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/project-vision.git
cd project-vision
```

### 2. Set Up Python Environment
Recommended configuration for local development (Linux / WSL2 with NVIDIA CUDA support):

```bash
# Create and activate conda environment
conda create -n project-vision python=3.10 -y
conda activate project-vision

# Install PyTorch with CUDA 11.8/12.1 support (adjust based on your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install neuromorphic processing and evaluation libraries
pip install tonic h5py matplotlib tqdm torchmetrics lpips pyyaml
```

---

## 🚀 Quickstart & Pipeline Verification

### Step 1: Run 2D Coordinate Sanity Check
Verify that your PyTorch coordinate network and positional encodings work by fitting a single test image:
```bash
python -m src.models.inr --test-toy-fit
```

### Step 2: Download Benchmark Data
Download a sample sequence from the public **DAVIS240C** dataset:
```bash
# Download sample event sequence to data/raw/
python -m src.data.dataset --download-sample shapes_rotation
```

### Step 3: Run Training (Local Debug Run)
Run a short sanity training cycle using mixed precision (`fp16`) configured for local development:
```bash
python train.py --config configs/base_inr.yaml --device cuda --epochs 5 --batch_size 16384
```

---

## 📊 Experimental Results & Benchmarks

<!-- ================================================================= -->
<!-- [UPDATE REQUIRED: As you run benchmarks, populate quantitative      -->
<!-- comparison metrics against SOTA baselines (E2VID, FireNet).]      -->
<!-- ================================================================= -->

### Quantitative Comparison on DAVIS240C & HQF

| Model Architecture | Paradigm | Time Binning | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | FPS / Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **E2VID** (Rebecq et al.) | Recurrent U-Net | Yes (Voxels) | *--* | *--* | *--* | *--* |
| **FireNet** (Scheerlinck et al.) | Fast Recurrent | Yes (Voxels) | *--* | *--* | *--* | *--* |
| **project-vision (Ours)** | Continuous INR | **None (Continuous)** | **[UPDATE]** | **[UPDATE]** | **[UPDATE]** | **[UPDATE]** |

> *Note: Evaluation calculated across the standard dynamic sequences of the DAVIS240C and HQF benchmarks.*

### Qualitative Comparisons

<!-- ================================================================= -->
<!-- [UPDATE REQUIRED: Insert comparative frame previews or GIFs here    -->
<!-- showing blurry ground truth vs sharp reconstructed frames]        -->
<!-- ================================================================= -->

| Blurry Frame (Standard CMOS) | Raw Asynchronous Events $(x, y, t)$ | Our Reconstruction (Continuous INR) |
| :---: | :---: | :---: |
| `[UPDATE: docs/assets/cmos_blur.png]` | `[UPDATE: docs/assets/events_raw.png]` | `[UPDATE: docs/assets/our_recon.png]` |

---

## 🔬 Core Methodology & Theoretical Innovations

<!-- ================================================================= -->
<!-- [UPDATE REQUIRED: Once you finalise your novelty (e.g. Fourier    -->
<!-- feature encoding variant, hash-grid acceleration, or zero-motion  -->
<!-- regularization loss), document the mathematical formulation here]  -->
<!-- ================================================================= -->

1. **Continuous Positional Encoding:** 
   $$\gamma(x, y, t) = \Big[\sin\big(2^0 \pi \mathbf{v}\big), \cos\big(2^0 \pi \mathbf{v}\big), \dots, \sin\big(2^{L-1} \pi \mathbf{v}\big), \cos\big(2^{L-1} \pi \mathbf{v}\big)\Big]$$
   where temporal coordinate $t$ is preserved continuously rather than collapsed into discrete tensor bins.

2. **Event-Consistency Differential Loss:**
   <!-- [UPDATE: Insert specific loss equation used to penalize drift during silence] -->
   $$\mathcal{L}_{total} = \mathcal{L}_{photometric} + \lambda_{e} \mathcal{L}_{event} + \lambda_{reg} \mathcal{L}_{drift}$$

---

## 🗓️ Research Roadmap & Milestones

- [x] Problem formulation and literature gap analysis
- [x] Environment configuration and dependency selection (`tonic`, `PyTorch`)
- [ ] **Milestone 1:** Build toy 2D coordinate MLP image-fitting pipeline
- [ ] **Milestone 2:** Ingest and normalize continuous DAVIS240C event streams
- [ ] **Milestone 3:** Reproduce E2VID / FireNet baselines on test sequences
- [ ] **Milestone 4:** Implement continuous spatio-temporal coordinate network
- [ ] **Milestone 5:** Formulate drift-suppression loss for zero-motion intervals
- [ ] **Milestone 6:** Full ablation suite on Kaggle GPU (T4 / 16GB VRAM)
- [ ] **Milestone 7:** Draft manuscript and release model checkpoints

---

## 💻 Hardware Execution Notes

* **Local Prototyping:** Tested on an NVIDIA RTX 3050 Laptop GPU (6GB VRAM, Intel i5-13420H, 16GB RAM). Local runs utilize PyTorch Automatic Mixed Precision (`torch.cuda.amp.autocast()`) for memory efficiency.
* **Full-Scale Cloud Benchmarking:** Extended multi-epoch training sweeps and full LPIPS perceptual loss evaluations are executed remotely using Kaggle NVIDIA T4 / P100 (16GB VRAM) compute sessions.

---

## 📜 Citation & Attribution

<!-- ================================================================= -->
<!-- [UPDATE REQUIRED: When you prepare your pre-print or paper,        -->
<!-- replace this placeholder with your formal BibTeX citation.]        -->
<!-- ================================================================= -->

If you use or build upon `project-vision` in your research, please cite:

```bibtex
@article{yourname2026projectvision,
  title={Continuous Event-Stream Frame Reconstruction via Implicit Neural Representations},
  author={Your Name},
  journal={Working Preprint / In Preparation for Q1 Venue},
  year={2026}
}
```

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.