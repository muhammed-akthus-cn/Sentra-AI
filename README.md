# Sentra AI — Edge-Based Crowd Crush Predictor

**Sentra AI** is a lightweight, offline-first Computer Vision pipeline engineered to detect early precursors of crowd crush and stampedes on constrained edge CPUs—without relying on cloud servers or heavy GPU hardware.

---

## 1. Engineering Problem & Architectural Pivot

Standard object detectors (such as `ssdlite320_mobilenet_v3_large`) fail in high-density public gatherings (>4 people/m²) due to **extreme body occlusion** and **Non-Maximum Suppression (NMS)** collapse—detecting only a handful of foreground individuals while missing hundreds packed behind them. Furthermore, static bounding boxes cannot measure crowd kinematics over time.

To solve this within edge CPU latency budgets, Sentra AI implements a **3-Stage Hybrid Pipeline**:

1. **ExecuTorch XNNPACK Base Detection (`ssdlite_xnnpack.pte`):** Runs quantized/portable MobileNetV3-SSDLite inference for baseline person detection and periodic calibration.
2. **2D Gaussian Crowd Density Field:** Extracts high-frequency head/shoulder micro-features via OpenCV's **FAST** detector and applies a $35 \times 35$ Gaussian kernel ($\sigma = 10.0$) mapped to a thermal risk scale (`COLORMAP_JET`), automatically outlining contiguous regions exceeding 70% density as **`CRITICAL CRUSH ZONES`**.
3. **Farneback Dense Optical Flow & Kinematic Risk Scoring:** Computes dense pixel-level motion vectors $(u, v)$ across active crowd regions to detect **Compressive Stagnation** (jamming) and **Multi-Directional Shockwave Turbulence**.

---

## 2. Mathematical Formulation

For each processed frame $t$ over the dense crowd mask $\Omega$:

* **Spatial Density Score ($D_t \in [0, 1]$):**
  $$D_t = \min\left(1.0, \; 0.65 \left(\frac{C_t}{C_{\text{crit}}}\right) + 0.35 \left(\frac{|\Omega|}{W \times H}\right)\right)$$

* **Compressive Stagnation Score ($S_t \in [0, 1]$):** Models immobilization risk when mean crowd velocity $V_t$ drops inside a high-density choke point:
  $$S_t = \exp\left(-\frac{V_t}{V_{\text{safe}}}\right)$$

* **Shockwave Turbulence Score ($T_t \in [0, 1]$):** Quantizes flow angles into $B=8$ directional bins and computes normalized **Shannon Angular Entropy ($H_t$)** weighted by velocity dispersion $\sigma_M$:
  $$H_t = -\frac{1}{\ln(8)} \sum_{i=1}^{8} p_i \ln(p_i), \qquad T_t = H_t \times \min\left(1.0, \frac{\sigma_M}{\sigma_{\text{crit}}}\right)$$

* **Unified Crush Risk Score ($CRS_t \in [0, 100]$):**
  $$CRS_t = D_t \times \max\left(0.95 \cdot S_t, \; 1.05 \cdot T_t\right) \times 100$$
  An Exponential Moving Average ($\alpha = 0.4$) smooths frame-to-frame jitter, triggering a **`CRUSH WARNING`** when $\overline{CRS}_t \ge 65.0$ and $D_t \ge 0.60$.

---

## 3. CPU Benchmark Results (720p HD Crowd Sequence)

Tested locally on an **AMD Ryzen 3 CPU** (Windows 11, ExecuTorch Portable/XNNPACK + OpenCV):

| Metric | Measured Result |
| :--- | :--- |
| **Input Resolution** | $1280 \times 720$ (25 FPS) |
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
