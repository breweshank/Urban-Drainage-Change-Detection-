import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import jaccard_score, f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ----------------------------
# Config
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 250
IMG_SIZE = 256
BEST_MODEL_PATH = "urban_drainage_unet_best.pth"
LAST_MODEL_PATH = "urban_drainage_unet_last.pth"

# ----------------------------
# Dataset Class
# ----------------------------
class HybridDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        mask = self.y[idx]
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        img = self.transform(img)  # (6,H,W)
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1,H,W)
        return img, mask

# ----------------------------
# Data Loader Function
# ----------------------------
def load_levir(path, target_size=(256,256)):
    X, Y = [], []
    for fname in sorted(os.listdir(os.path.join(path,"A"))):
        imgA = cv2.imread(os.path.join(path,"A",fname))
        imgB = cv2.imread(os.path.join(path,"B",fname))
        label = cv2.imread(os.path.join(path,"label",fname), cv2.IMREAD_GRAYSCALE)
        if imgA is None or imgB is None or label is None:
            continue
        imgA = cv2.resize(imgA, target_size) / 255.0
        imgB = cv2.resize(imgB, target_size) / 255.0
        label = (cv2.resize(label, target_size, interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
        stacked = np.concatenate([imgA, imgB], axis=-1)  # 6 channels
        X.append(stacked.astype(np.float32))
        Y.append(label.astype(np.float32))
    return np.array(X), np.array(Y)

# ----------------------------
# UNet Architecture
# ----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(UNet, self).__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

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

# ----------------------------
# Metrics
# ----------------------------
def calc_iou_f1(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return jaccard_score(y_true, y_pred), f1_score(y_true, y_pred)

# ----------------------------
# Training
# ----------------------------
def train_model():
    # Dataset paths
    levir_train_path = r"C:\Users\ESHANK\Downloads\ISRO\Eshank\archive (3)\LEVIR CD\train"
    levir_val_path   = r"C:\Users\ESHANK\Downloads\ISRO\Eshank\archive (3)\LEVIR CD\val"
    levir_test_path  = r"C:\Users\ESHANK\Downloads\ISRO\Eshank\archive (3)\LEVIR CD\test"

    # Load data
    X_train, y_train = load_levir(levir_train_path, (IMG_SIZE, IMG_SIZE))
    X_val, y_val = load_levir(levir_val_path, (IMG_SIZE, IMG_SIZE))
    X_test, y_test = load_levir(levir_test_path, (IMG_SIZE, IMG_SIZE))

    # Dataloaders
    train_loader = DataLoader(HybridDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(HybridDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(HybridDataset(X_test, y_test), batch_size=BATCH_SIZE)

    # Model, Loss, Optimizer
    model = UNet(in_channels=6, out_channels=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_iou = 0.0

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_gts = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE)
                outputs = torch.sigmoid(model(imgs)).cpu().numpy()
                preds = (outputs > 0.5).astype(np.uint8)
                val_preds.append(preds)
                val_gts.append(masks.numpy())

        val_preds = np.concatenate(val_preds).flatten()
        val_gts = np.concatenate(val_gts).flatten()

        val_iou, val_f1 = calc_iou_f1(val_gts, val_preds)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {train_loss/len(train_loader):.4f} Val IoU: {val_iou:.4f} Val F1: {val_f1:.4f}")

        # Save best
        if val_iou > best_iou:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_iou = val_iou
            print(f"Saved best model with IoU {best_iou:.4f}")

        # Always save last
        torch.save(model.state_dict(), LAST_MODEL_PATH)

    print("Training complete.")

    # Testing
    if len(X_test) > 0:
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        model.eval()
        test_preds, test_gts = [], []
        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs = imgs.to(DEVICE)
                outputs = torch.sigmoid(model(imgs)).cpu().numpy()
                preds = (outputs > 0.5).astype(np.uint8)
                test_preds.append(preds)
                test_gts.append(masks.numpy())
        test_preds = np.concatenate(test_preds).flatten()
        test_gts = np.concatenate(test_gts).flatten()

        test_iou, test_f1 = calc_iou_f1(test_gts, test_preds)
        print(f"TEST IoU: {test_iou:.4f} | TEST F1: {test_f1:.4f}")

if __name__ == "__main__":
    train_model()

