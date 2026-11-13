# Sentra AI — Edge-Based Crowd Crush Predictor

**Sentra AI** is a lightweight, offline-first Computer Vision pipeline engineered to detect early precursors of crowd crush and stampedes on constrained edge CPUs—without relying on cloud servers or heavy GPU hardware.

---

## 1. Engineering Problem & Architectural Pivot

Standard object detectors (such as `ssdlite320_mobilenet_v3_large`) fail in high-density public gatherings (>4 people/m²) due to **extreme body occlusion** and **Non-Maximum Suppression (NMS)** collapse—detecting only a handful of foreground individuals while missing hundreds packed behind them. Furthermore, static bounding boxes cannot measure crowd kinematics over time.

To solve this within edge CPU latency budgets, Sentra AI implements a **3-Stage Hybrid Pipeline**:

1. **ExecuTorch XNNPACK Base Detection (`ssdlite_xnnpack.pte`):** Runs quantized/portable MobileNetV3-SSDLite inference for baseline person detection and periodic calibration.
2. **2D Gaussian Crowd Density Field:** Extracts high-frequency head/shoulder micro-features via OpenCV's **FAST** detector and applies a **35 × 35** Gaussian kernel (**σ = 10.0**) mapped to a thermal risk scale (`COLORMAP_JET`), automatically outlining contiguous regions exceeding 70% density as `CRITICAL CRUSH ZONES`.
3. **Farneback Dense Optical Flow & Kinematic Risk Scoring:** Computes dense pixel-level motion vectors **(u, v)** across active crowd regions to detect **Compressive Stagnation** (jamming) and **Multi-Directional Shockwave Turbulence**.

---

## 2. Mathematical Formulation

For each processed frame **t** over the dense crowd mask **Ω**:

* **Spatial Density Score (D<sub>t</sub> ∈ [0, 1]):**
  > **D<sub>t</sub> = min( 1.0,  0.65 · (C<sub>t</sub> / C<sub>crit</sub>) + 0.35 · (|Ω| / (W × H)) )**

* **Compressive Stagnation Score (S<sub>t</sub> ∈ [0, 1]):** Models immobilization risk when mean crowd velocity **V<sub>t</sub>** drops inside a high-density choke point:
  > **S<sub>t</sub> = exp( −V<sub>t</sub> / V<sub>safe</sub> )**

* **Shockwave Turbulence Score (T<sub>t</sub> ∈ [0, 1]):** Quantizes flow angles into **B = 8** directional bins and computes normalized **Shannon Angular Entropy (H<sub>t</sub>)** weighted by velocity dispersion **σ<sub>M</sub>**:
  > **H<sub>t</sub> = −(1 / ln 8) · ∑<sub>i=1</sub><sup>8</sup> p<sub>i</sub> ln(p<sub>i</sub>)**
  >
  > **T<sub>t</sub> = H<sub>t</sub> · min( 1.0,  σ<sub>M</sub> / σ<sub>crit</sub> )**

* **Unified Crush Risk Score (CRS<sub>t</sub> ∈ [0, 100]):**
  > **CRS<sub>t</sub> = D<sub>t</sub> × max( 0.95 · S<sub>t</sub>,  1.05 · T<sub>t</sub> ) × 100**

  An Exponential Moving Average (**α = 0.4**) smooths frame-to-frame jitter, triggering a `CRUSH WARNING` when the smoothed score **CRS<sub>avg</sub> ≥ 65.0** and **D<sub>t</sub> ≥ 0.60**.

---

## 3. CPU Benchmark Results (720p HD Crowd Sequence)

Tested locally on an **AMD Ryzen 3 CPU** (Windows 11, ExecuTorch Portable/XNNPACK + OpenCV):

| Metric | Measured Result |
| :--- | :--- |
| **Input Resolution** | 1280 × 720 (25 FPS) |
| **Detected Crowd Count (Peak)** | 220 – 384 people (vs. ~10 via raw SSD boxes) |
| **Optical Flow + Density Latency** | **29.9 ms / frame (~33.4 FPS)** |
| **ExecuTorch Sync Frame Latency** | **183.8 ms / frame** |
| **Network Requirement** | 0 Mbps (100% offline; supports local RTSP CCTV streams) |

---

## 4. Quick Start

```bash
# Install dependencies
pip install torch torchvision executorch opencv-python numpy

# Run on an image, image folder, or .mp4 / RTSP CCTV stream
python sentra_final.py
