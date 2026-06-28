# End-to-End Automated AAA Wall Strain Mapping

> Dense Optical Flow + U-Net Segmentation for Abdominal Aortic Aneurysm Biomechanics

---

## 🔍 Background

Non-invasive biomechanical characterisation of **abdominal aortic aneurysm (AAA)** wall tissue requires accurate segmentation and motion tracking on time-resolved B-mode ultrasound (US) sequences.

Existing methods rely on:
- **Manual contour delineation** a bottleneck that prevents clinical scalability
- **Sparse feature-point tracking**  which degrades in echogenically heterogeneous regions

---

## ⚙️ Methods

A fully automated, end-to-end pipeline built entirely on **open-source data and tooling**:

| Step | Component |
|------|-----------|
| Segmentation | U-Net trained on [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) + simulated AAA frames |
| Tracking | Dense bilateral TV-L1 Optical Flow (replaces Sparse Demons) |
| Strain computation | Radial Basis Function (RBF) interpolation |

---

## 📊 Results

| Metric | Baseline (Manual + SD) | Proposed (U-Net + Dense OF) |
|--------|------------------------|------------------------------|
| Seg. Dice | — | **0.908 ± 0.021** |
| Seg. IoU | — | **0.849 ± 0.028** |
| Track RMSE | 0.142 ± 0.058 mm | **0.087 ± 0.031 mm** |
| Strain MAE | 1.81% | **0.93%** |
| Time / frame | ~180 ms | **~68 ms** |

> Tracking improvement: **−39% RMSE** (p < 0.05, Wilcoxon signed-rank test)

---

## ✅ Conclusion

The proposed pipeline:
- Removes the **manual segmentation bottleneck**
- Improves tracking in **low-echogenicity lateral wall regions**
- Achieves **near real-time throughput** (~68 ms/frame)
- Uses exclusively **open-source datasets and software**

---

## 🏷️ Keywords

`abdominal aortic aneurysm` `ultrasound` `wall segmentation` `U-Net` `dense optical flow` `strain imaging` `biomechanics`