import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
MODEL_PATH = r"C:\Users\ESHANK\Downloads\ISRO\Eshank\urban_drainage_unet_best.pth"

# ---------------- UNET MODEL ----------------
class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet(torch.nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = torch.nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = torch.nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = torch.nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = torch.nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)

# ---------------- LOAD MODEL ----------------
model = UNet(in_channels=6, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- HELPER ----------------
def load_and_preprocess(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    return img.astype(np.float32)

def predict_change(imgA_path, imgB_path, mask_path=None):
    imgA = load_and_preprocess(imgA_path)
    imgB = load_and_preprocess(imgB_path)

    combined = np.concatenate([imgA, imgB], axis=-1)  # 6 channels
    combined = torch.from_numpy(combined.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(model(combined)).cpu().numpy()[0, 0]
        mask = (pred > 0.5).astype(np.uint8)

    overlay = imgB.copy()
    overlay[mask == 1] = [255, 0, 0]  # Red = change

    iou, f1 = None, None
    if mask_path:
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (IMG_SIZE, IMG_SIZE))
        gt = (gt > 127).astype(np.uint8)
        iou = jaccard_score(gt.flatten(), mask.flatten())
        f1 = f1_score(gt.flatten(), mask.flatten())

    return imgA, imgB, overlay, iou, f1

# ---------------- GUI ----------------
class ChangeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Urban Drainage Change Detection")
        self.root.geometry("1000x700")
        self.root.configure(bg="#f0f0f0")

        self.imgA_path = None
        self.imgB_path = None
        self.mask_path = None

        tk.Label(root, text="Urban Drainage Change Detection",
                 font=("Helvetica", 18, "bold"), bg="#f0f0f0").pack(pady=10)

        btn_frame = tk.Frame(root, bg="#f0f0f0")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Upload BEFORE Image", command=self.upload_before,
                  width=20, bg="#2196F3", fg="white").grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="Upload AFTER Image", command=self.upload_after,
                  width=20, bg="#4CAF50", fg="white").grid(row=0, column=1, padx=10)
        tk.Button(btn_frame, text="Upload Mask (Optional)", command=self.upload_mask,
                  width=20, bg="#9C27B0", fg="white").grid(row=0, column=2, padx=10)

        tk.Button(root, text="Run Prediction", command=self.run_prediction,
                  width=25, bg="#FF5722", fg="white", font=("Helvetica", 12, "bold")).pack(pady=15)

        self.canvas = tk.Label(root)
        self.canvas.pack()

        self.result_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#f0f0f0")
        self.result_label.pack(pady=10)

    def upload_before(self):
        self.imgA_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif")])
        messagebox.showinfo("Uploaded", "Before image uploaded successfully!")

    def upload_after(self):
        self.imgB_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif")])
        messagebox.showinfo("Uploaded", "After image uploaded successfully!")

    def upload_mask(self):
        self.mask_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif")])
        messagebox.showinfo("Uploaded", "Mask uploaded successfully!")

    def run_prediction(self):
        if not self.imgA_path or not self.imgB_path:
            messagebox.showerror("Error", "Please upload both BEFORE and AFTER images.")
            return

        imgA, imgB, overlay, iou, f1 = predict_change(self.imgA_path, self.imgB_path, self.mask_path)

        # Combine results in one figure
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        ax[0].imshow(imgA); ax[0].set_title("Before")
        ax[1].imshow(imgB); ax[1].set_title("After")
        ax[2].imshow(overlay); ax[2].set_title("Change")
        for a in ax: a.axis('off')
        fig.tight_layout()

        # Save and display in GUI
        fig.savefig("temp_result.png")
        plt.close(fig)

        img = Image.open("temp_result.png")
        img = img.resize((800, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.config(image=img_tk)
        self.canvas.image = img_tk

        # Show metrics
        if iou is not None and f1 is not None:
            self.result_label.config(text=f"IoU: {iou:.4f} | F1 Score: {f1:.4f}")
        else:
            self.result_label.config(text="No ground truth mask provided.")

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ChangeDetectionApp(root)
    root.mainloop()

