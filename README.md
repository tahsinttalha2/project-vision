# Continuous Event-Stream Video Reconstruction via Implicit Neural Representations

[![Status: Work in Progress](https://img.shields.io/badge/Status-Work_in_Progress-yellow.svg)](#)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](#)

## Overview

Event cameras (neuromorphic vision sensors) measure asynchronous per-pixel brightness changes with microsecond latency and high dynamic range (>120 dB), yielding sparse spatio-temporal event streams rather than synchronous 2D frames. 

While existing learning-based reconstruction methods (e.g., E2VID, FireNet) typically discretise continuous events into fixed 2D/3D voxel grids, this process incurs temporal aliasing and discards exact microsecond timestamps. 

This repository investigates **continuous coordinate representations** for intensity video reconstruction. By parameterising the latent spatio-temporal radiance field as a continuous function $f_\theta(x, y, t) \rightarrow I$, this approach aims to:
- Preserve microsecond temporal resolution without temporal voxel binning.
- Mitigate recurrent state drift and ghosting artefacts during low-motion periods.
- Enable arbitrary-framerate video synthesis directly from asynchronous event streams.

---

## Mathematical Formulation

An event stream is defined as an asynchronous sequence:

$$\mathcal{E} = \{e_k\}_{k=1}^N, \quad e_k = (x_k, y_k, t_k, p_k)$$

where:
- $(x_k, y_k) \in [0, W-1] \times [0, H-1]$ denotes the pixel coordinates.
- $t_k \in \mathbb{R}^+$ is the microsecond timestamp.
- $p_k \in \{-1, +1\}$ is the polarity representing the sign of log-intensity change.

The network approximates the latent intensity function $I(x, y, t)$ continuously over the spatio-temporal domain via continuous frequency embeddings and coordinate-based neural layers.

---

## Project Status

> **Note:** This project is actively under development. Codebases, model architectures, and experimental pipelines are currently being prototyped and refined.

### Planned Methodology
- [ ] Data preprocessing and parsing pipeline using the `Tonic` neuromorphic framework.
- [ ] Continuous spatio-temporal coordinate encoding (Fourier / Hash features).
- [ ] Unsupervised / self-supervised event consistency loss formulation.
- [ ] Quantitative benchmarking against baseline models on DAVIS240C and MVSEC datasets.

---

## Primary References

- **E2VID:** Rebecq, H., Ranftl, R., Koltun, V., & Scaramuzza, D. (2021). *High Speed and High Dynamic Range Video with an Event Camera*. IEEE TPAMI.
- **FireNet:** Scheerlinck, C., Rebecq, H., Gehrig, D., et al. (2020). *Fast Image Reconstruction with an Event Camera*. IEEE WACV.
- **Tonic:** Neuromorphic dataset loading and event streaming library.
