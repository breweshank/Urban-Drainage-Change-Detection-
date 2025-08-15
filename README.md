# 🛰 Urban Drainage Change Detection (UNet + Tkinter GUI)

A **deep learning-powered GUI tool** for detecting changes between *Before* and *After* satellite/drone images  
of urban drainage systems. Built with **PyTorch** for the model and **Tkinter** for a simple, user-friendly interface.  

---

## 📖 Background

Urban drainage networks are critical for:
- Managing stormwater
- Preventing floods
- Maintaining public safety

Climate change, unplanned urban expansion, and aging infrastructure can cause **unexpected changes** in drainage networks.  
Traditionally, engineers manually compare images to identify these changes — a process that is slow, subjective, and prone to errors.

This project automates the detection process using **semantic segmentation** with a **custom 6-channel UNet model**.

---

## 🧪 Methodology

**Step-by-step pipeline:**

1. **Data Preparation**
   - Collect paired *Before* and *After* RGB images of the same area.
   - Create binary masks marking changed areas (ground truth).
   - Resize images to **256×256** for uniformity.

2. **Model Architecture**
   - **Input**: 6 channels (Before RGB + After RGB).
   - **Architecture**: UNet with skip connections, batch normalization, and ReLU activation.
   - **Output**: Single-channel binary mask (change = 1, no change = 0).

3. **Inference**
   - Upload Before and After images via GUI.
   - Concatenate into a 6-channel tensor.
   - Pass through UNet.
   - Threshold output at `0.5` to get binary change mask.
   - Overlay change regions on After image.

4. **Evaluation**
   - If a ground truth mask is provided:
     - **IoU** (Intersection over Union)
     - **F1-score** (Dice Similarity)

---
## **Project Strcuture**
```
urban_drainage_change_detection/
│
├── dataset/│
|        |___before/    # Before images
|        ├___ after/     # After images
|        └── masks/     # Binary masks
├── train.py
├── test.py
├── requirements.txt                   # Dependencies
├── README.md                          # Documentation
└── docs/
    ├── example_before.jpg
    ├── example_after.jpg
    ├── example_overlay.jpg
```
