"""
aaa_strain_pipeline.py
======================
End-to-End Automated Pipeline for AAA Wall Strain Mapping
as described in the accompanying paper.

Dependencies:
    pip install torch torchvision numpy scipy scikit-image opencv-python matplotlib tqdm

Usage:
    # Train the U-Net
    python aaa_strain_pipeline.py --mode train --data_dir /path/to/data

    # Run inference on a sequence
    python aaa_strain_pipeline.py --mode infer --weights best_model.pth \
                                  --sequence /path/to/sequence.npy

    # Generate synthetic training data
    python aaa_strain_pipeline.py --mode simulate --n_sequences 80 --out_dir /path/to/out

License: MIT
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RBFInterpolator
from skimage.draw import ellipse as sk_ellipse
from skimage.measure import find_contours
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# 1. U-NET ARCHITECTURE
# ===========================================================================

class DoubleConv(nn.Module):
    """Two consecutive Conv2d + BatchNorm + ReLU blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Standard U-Net for AAA wall segmentation.

    Input : (B, 1, 256, 256) grayscale US frame
    Output: (B, 1, 256, 256) soft wall probability mask
    """
    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.enc = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.enc.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        self.up_convs  = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        dec_features = list(reversed(features))
        ch = features[-1] * 2
        for f in dec_features:
            self.up_convs.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.dec_blocks.append(DoubleConv(f * 2, f))
            ch = f

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        enc_features = []
        for enc_block in self.enc:
            x = enc_block(x)
            enc_features.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.up_convs, self.dec_blocks, reversed(enc_features)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = dec(torch.cat([skip, x], dim=1))

        return torch.sigmoid(self.final(x))


# ===========================================================================
# 2. LOSS FUNCTION
# ===========================================================================

class BCEDiceLoss(nn.Module):
    """Combined BCE + soft Dice loss (Eq. 1 in the paper)."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.bce = nn.BCELoss()
        self.eps = eps

    def forward(self, pred, target):
        bce_val = self.bce(pred, target)
        # Soft Dice
        inter = (pred * target).sum(dim=(1,2,3))
        union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice_val = 1 - (2 * inter + self.eps) / (union + self.eps)
        return bce_val + dice_val.mean()


# ===========================================================================
# 3. SYNTHETIC AAA DATASET
# ===========================================================================

class SyntheticAAADataset(Dataset):
    """
    Generates (or loads pre-generated) simulated AAA US frames with masks.

    Each sample is a dictionary:
        'image' : (1, 256, 256) float32 tensor in [0,1]
        'mask'  : (1, 256, 256) float32 binary tensor
    """
    def __init__(self, n_samples=200, img_size=256, augment=True, seed=42):
        np.random.seed(seed)
        self.n_samples = n_samples
        self.img_size  = img_size
        self.augment   = augment
        self.samples   = [self._generate() for _ in range(n_samples)]

    def _generate(self):
        S = self.img_size
        # Random ellipse parameters
        ry = int(np.random.uniform(0.18, 0.29) * S)   # minor radius (AP)
        rx = int(np.random.uniform(0.18, 0.32) * S)   # major radius (LR)
        cy = int(np.random.uniform(0.42, 0.58) * S)
        cx = int(np.random.uniform(0.42, 0.58) * S)
        wall_t = int(np.random.uniform(0.020, 0.030) * S)  # 5-8 px

        # Background speckle (exponential distribution → Rayleigh-like envelope)
        img = np.random.exponential(0.14, (S, S)).astype(np.float32)
        img = gaussian_filter(img, sigma=1.4)

        # Horizontal striping (US beam pattern)
        for i in range(0, S, 7):
            img[i:i+2, :] *= np.random.uniform(0.6, 0.85)

        # Lumen: dark ellipse
        rr, cc = sk_ellipse(cy, cx, ry, rx, shape=(S, S))
        lumen_m = np.zeros((S, S), bool); lumen_m[rr, cc] = True
        img[lumen_m] = np.random.exponential(0.025, lumen_m.sum())

        # Wall ring: bright
        rr2, cc2 = sk_ellipse(cy, cx, ry+wall_t, rx+wall_t, shape=(S, S))
        wall_m = np.zeros((S, S), bool); wall_m[rr2, cc2] = True
        ring = wall_m & ~lumen_m
        img[ring] = np.random.uniform(0.55, 0.90, ring.sum()).astype(np.float32)

        img = np.clip(img, 0, 1)
        mask = ring.astype(np.float32)
        return img[None], mask[None]   # (1, S, S)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img, mask = [torch.from_numpy(x.copy()) for x in self.samples[idx]]
        if self.augment:
            img, mask = self._augment(img, mask)
        return {'image': img, 'mask': mask}

    def _augment(self, img, mask):
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            img  = torch.flip(img,  dims=[2])
            mask = torch.flip(mask, dims=[2])
        # Random vertical flip
        if torch.rand(1) > 0.5:
            img  = torch.flip(img,  dims=[1])
            mask = torch.flip(mask, dims=[1])
        # Brightness jitter
        img = (img * torch.empty(1).uniform_(0.85, 1.15)).clamp(0, 1)
        # Additive noise
        img = (img + torch.randn_like(img) * 0.03).clamp(0, 1)
        return img, mask


# ===========================================================================
# 4. TRAINING LOOP
# ===========================================================================

def train_unet(data_dir=None, out_dir='./weights', n_epochs=100,
               batch_size=8, lr=1e-4, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = SyntheticAAADataset(n_samples=600, augment=True,  seed=0)
    val_ds   = SyntheticAAADataset(n_samples=100, augment=False, seed=42)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    model = UNet().to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, verbose=True)

    best_dice = 0.0
    history = {'train_loss':[], 'val_loss':[], 'val_dice':[], 'val_iou':[]}

    for epoch in range(1, n_epochs+1):
        # ── Train ──
        model.train()
        t_loss = 0
        for batch in tqdm(train_dl, desc=f'Epoch {epoch}/{n_epochs}', leave=False):
            imgs  = batch['image'].to(device)
            masks = batch['mask'].to(device)
            preds = model(imgs)
            loss  = criterion(preds, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_dl)

        # ── Validate ──
        model.eval()
        v_loss = v_dice = v_iou = 0
        with torch.no_grad():
            for batch in val_dl:
                imgs  = batch['image'].to(device)
                masks = batch['mask'].to(device)
                preds = model(imgs)
                v_loss += criterion(preds, masks).item()
                pbin  = (preds > 0.5).float()
                inter = (pbin * masks).sum(dim=(1,2,3))
                union = pbin.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3))
                v_dice += (2*inter / (union + 1e-6)).mean().item()
                v_iou  += (inter / (union - inter + 1e-6)).mean().item()
        v_loss /= len(val_dl); v_dice /= len(val_dl); v_iou /= len(val_dl)

        scheduler.step(v_loss)
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_dice'].append(v_dice)
        history['val_iou'].append(v_iou)

        print(f"Epoch {epoch:3d} | Train Loss: {t_loss:.4f} | "
              f"Val Loss: {v_loss:.4f} | Dice: {v_dice:.4f} | IoU: {v_iou:.4f}")

        if v_dice > best_dice:
            best_dice = v_dice
            torch.save(model.state_dict(), f'{out_dir}/best_model.pth')
            print(f"  ✓ Saved best model (Dice={best_dice:.4f})")

    return model, history


# ===========================================================================
# 5. DENSE OPTICAL FLOW TRACKING
# ===========================================================================

class DenseOpticalFlowTracker:
    """
    Bilateral TV-L1 dense optical flow tracker (Zach et al., 2007).

    Uses OpenCV's DualTVL1OpticalFlow implementation.
    """
    def __init__(self, lambda_=0.15, theta=0.3, n_scales=5, n_iters=300):
        self.flow_estimator = cv2.optflow.DualTVL1OpticalFlow_create(
            lambda_=lambda_,
            theta=theta,
            nscales=n_scales,
            warps=5,
            tau=0.25,
            epsilon=0.01,
            innnerIterations=n_iters,
            outerIterations=10,
            scaleStep=0.5,
            gamma=0.0,
            useInitialFlow=False,
        )

    def compute_flow(self, frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
        """
        Compute dense displacement field from frame0 to frame1.

        Parameters
        ----------
        frame0, frame1 : np.ndarray, shape (H, W), float32 in [0,1]

        Returns
        -------
        flow : np.ndarray, shape (H, W, 2), float32 — (u, v) in pixels
        """
        f0 = (frame0 * 255).astype(np.uint8)
        f1 = (frame1 * 255).astype(np.uint8)
        flow = self.flow_estimator.calc(f0, f1, None)
        return flow  # (H, W, 2)

    def track_sequence(self, frames: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Track all pixels within mask over a sequence of frames.

        Parameters
        ----------
        frames : (T, H, W) float32
        mask   : (H, W) bool — ROI mask

        Returns
        -------
        disp : (T, H, W, 2) float32 — cumulative displacement from frame 0
        """
        T, H, W = frames.shape
        disp = np.zeros((T, H, W, 2), dtype=np.float32)
        cumul = np.zeros((H, W, 2), dtype=np.float32)

        for t in range(1, T):
            flow = self.compute_flow(frames[t-1], frames[t])
            flow[~mask] = 0.0
            cumul += flow
            disp[t] = cumul.copy()

        return disp


# ===========================================================================
# 6. CARDIAC CYCLE DETECTION
# ===========================================================================

def detect_cardiac_cycles(seg_masks: np.ndarray):
    """
    Detect end-diastole (minima) and peak-systole (maxima) from the
    antero-posterior diameter curve extracted from segmentation masks.

    Parameters
    ----------
    seg_masks : (T, H, W) binary masks

    Returns
    -------
    ap_diameters : (T,) array of AP diameter in pixels
    systole_idx  : list of peak-systole frame indices
    diastole_idx : list of end-diastole frame indices
    """
    from scipy.signal import find_peaks

    T, H, W = seg_masks.shape
    ap_diams = np.zeros(T)

    for t in range(T):
        m = seg_masks[t]
        rows = np.where(m.any(axis=1))[0]
        if len(rows) > 0:
            ap_diams[t] = rows[-1] - rows[0]

    # Smooth the AP curve
    ap_smooth = gaussian_filter(ap_diams, sigma=1.5)

    # Find peaks (systole) and troughs (diastole)
    sys_idx, _ = find_peaks(ap_smooth, distance=10, prominence=1.0)
    dia_idx, _ = find_peaks(-ap_smooth, distance=10, prominence=1.0)

    return ap_diams, list(sys_idx), list(dia_idx)


# ===========================================================================
# 7. RBF STRAIN COMPUTATION
# ===========================================================================

def compute_rbf_strain(disp_field: np.ndarray, wall_contour: np.ndarray,
                       n_theta: int = 80, n_layers: int = 4,
                       wall_thickness_px: float = 12.0):
    """
    Compute circumferential (hoop) strain on the AAA wall using RBF interpolation.

    Parameters
    ----------
    disp_field    : (H, W, 2) displacement field (u, v) in pixels
    wall_contour  : (N, 2) array of (row, col) wall contour points
    n_theta       : number of circumferential sampling stations
    n_layers      : number of transmural layers
    wall_thickness_px : wall thickness in pixels

    Returns
    -------
    strain_theta  : (n_theta,) circumferential strain values
    grid_pts      : (n_theta * n_layers, 2) grid point positions
    """
    # Fit ellipse to contour to get centroid and tangent directions
    from skimage.measure import EllipseModel
    model = EllipseModel()
    model.estimate(wall_contour)
    xc, yc, a, b, theta_ell = model.params

    # Generate sampling grid: annular from inner to outer wall
    angles = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    grid_pts = []
    for layer_frac in np.linspace(0, 1, n_layers):
        r = 1.0 + layer_frac * wall_thickness_px / max(a, b)
        for ang in angles:
            row = yc + r * b * np.sin(ang)
            col = xc + r * a * np.cos(ang)
            grid_pts.append([row, col])
    grid_pts = np.array(grid_pts)

    # Sample displacement at grid points using RBF fitted to dense flow
    H, W = disp_field.shape[:2]
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    src_pts = np.stack([ys.ravel(), xs.ravel()], axis=1)
    u_vals  = disp_field[:,:,0].ravel()
    v_vals  = disp_field[:,:,1].ravel()

    # Sub-sample for speed (every 4th pixel)
    idx = np.arange(0, len(src_pts), 4)
    rbf_u = RBFInterpolator(src_pts[idx], u_vals[idx], kernel='thin_plate_spline')
    rbf_v = RBFInterpolator(src_pts[idx], v_vals[idx], kernel='thin_plate_spline')

    u_grid = rbf_u(grid_pts)
    v_grid = rbf_v(grid_pts)

    # Circumferential strain: ε_θθ = dl/l₀ along arc
    strain_theta = np.zeros(n_theta)
    for i in range(n_theta):
        j = (i + 1) % n_theta
        # Points at mid-layer
        mid = n_layers // 2
        idx_i = mid * n_theta + i
        idx_j = mid * n_theta + j
        p0 = grid_pts[idx_i];  p1 = grid_pts[idx_j]
        d0 = np.array([u_grid[idx_i], v_grid[idx_i]])
        d1 = np.array([u_grid[idx_j], v_grid[idx_j]])
        l0 = np.linalg.norm(p1 - p0) + 1e-9
        l1 = np.linalg.norm((p1 + d1) - (p0 + d0)) + 1e-9
        strain_theta[i] = (l1 - l0) / l0

    return strain_theta, grid_pts


# ===========================================================================
# 8. FULL INFERENCE PIPELINE
# ===========================================================================

class AAAStrainPipeline:
    """
    End-to-end pipeline: US sequence → circumferential strain map.
    """
    def __init__(self, model_weights: str, device='cpu'):
        self.device = torch.device(device)
        self.model  = UNet().to(self.device)
        self.model.load_state_dict(
            torch.load(model_weights, map_location=self.device))
        self.model.eval()
        self.tracker = DenseOpticalFlowTracker()

    @torch.no_grad()
    def segment_sequence(self, frames: np.ndarray) -> np.ndarray:
        """
        Segment every frame in the sequence.

        Parameters
        ----------
        frames : (T, H, W) float32

        Returns
        -------
        masks : (T, H, W) bool
        """
        T, H, W = frames.shape
        masks = np.zeros((T, H, W), bool)
        for t in range(T):
            x = torch.from_numpy(frames[t][None, None]).to(self.device)
            prob = self.model(x)[0, 0].cpu().numpy()
            masks[t] = prob > 0.5
        return masks

    def run(self, frames: np.ndarray):
        """
        Full pipeline.

        Parameters
        ----------
        frames : (T, H, W) float32 B-mode sequence in [0,1]

        Returns
        -------
        result : dict with keys 'masks', 'ap_diams', 'strain', 'grid_pts'
        """
        print("Step 1: Segmentation...")
        masks = self.segment_sequence(frames)

        print("Step 2: Cardiac cycle detection...")
        ap_diams, sys_idx, dia_idx = detect_cardiac_cycles(masks)

        # Use first complete cycle
        if len(dia_idx) >= 2:
            t_start, t_end = dia_idx[0], dia_idx[1]
        else:
            t_start, t_end = 0, len(frames) - 1

        cycle_frames = frames[t_start:t_end+1]
        cycle_mask   = masks[t_start].astype(bool)

        print("Step 3: Dense optical flow tracking...")
        disp_seq = self.tracker.track_sequence(cycle_frames, cycle_mask)

        # Use displacement at peak systole relative to start
        if len(sys_idx) > 0:
            sys_local = [s - t_start for s in sys_idx if t_start <= s <= t_end]
            peak_t = sys_local[0] if sys_local else len(cycle_frames)//2
        else:
            peak_t = len(cycle_frames) // 2

        disp_peak = disp_seq[peak_t]

        print("Step 4: RBF strain computation...")
        # Get wall contour from first-frame mask
        contours = find_contours(cycle_mask.astype(float), 0.5)
        if not contours:
            raise ValueError("No wall contour found in segmentation mask.")
        contour = max(contours, key=len)

        strain, grid_pts = compute_rbf_strain(disp_peak, contour)

        print("Done.")
        return {
            'masks'     : masks,
            'ap_diams'  : ap_diams,
            'sys_idx'   : sys_idx,
            'dia_idx'   : dia_idx,
            'disp_peak' : disp_peak,
            'strain'    : strain,
            'grid_pts'  : grid_pts,
            'contour'   : contour,
        }

    @staticmethod
    def visualise(frame: np.ndarray, result: dict, out_path: str = 'strain_map.png'):
        """Overlay strain map on B-mode frame."""
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(frame, cmap='gray', vmin=0, vmax=1)

        strain   = result['strain']
        grid_pts = result['grid_pts']
        n_theta  = len(strain)
        n_layers = len(grid_pts) // n_theta

        segs, cols = [], []
        mid = n_layers // 2
        for i in range(n_theta):
            j = (i+1) % n_theta
            p0 = grid_pts[mid*n_theta + i][[1,0]]  # (col, row)
            p1 = grid_pts[mid*n_theta + j][[1,0]]
            segs.append([p0, p1])
            cols.append(strain[i])

        lc = LineCollection(segs, cmap='RdYlGn_r', linewidths=5)
        lc.set_array(np.array(cols)); lc.set_clim(0, 0.15)
        ax.add_collection(lc)
        cb = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.04)
        cb.set_label('Circumferential strain ε', fontsize=10)
        ax.axis('off')
        ax.set_title('AAA Wall Circumferential Strain Map', fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Strain map saved to {out_path}")


# ===========================================================================
# 9. EVALUATION METRICS
# ===========================================================================

def dice_score(pred: np.ndarray, gt: np.ndarray, eps=1e-6) -> float:
    inter = (pred & gt).sum()
    return float(2 * inter / (pred.sum() + gt.sum() + eps))

def iou_score(pred: np.ndarray, gt: np.ndarray, eps=1e-6) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / (union + eps))

def hausdorff_95(pred: np.ndarray, gt: np.ndarray) -> float:
    from scipy.spatial.distance import directed_hausdorff
    pred_pts = np.argwhere(pred)
    gt_pts   = np.argwhere(gt)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float('inf')
    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    return max(d1, d2)

def tracking_rmse(pred_disp: np.ndarray, gt_disp: np.ndarray,
                  mask: np.ndarray) -> float:
    err = np.linalg.norm(pred_disp[mask] - gt_disp[mask], axis=-1)
    return float(np.sqrt(np.mean(err**2)))

def strain_mae(pred_strain: np.ndarray, gt_strain: np.ndarray) -> float:
    return float(np.mean(np.abs(pred_strain - gt_strain)))


# ===========================================================================
# 10. MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='AAA Wall Strain Pipeline')
    parser.add_argument('--mode', choices=['train', 'infer', 'simulate', 'demo'],
                        default='demo')
    parser.add_argument('--data_dir',   default='./data')
    parser.add_argument('--out_dir',    default='./weights')
    parser.add_argument('--weights',    default='./weights/best_model.pth')
    parser.add_argument('--sequence',   default=None)
    parser.add_argument('--n_sequences', type=int, default=80)
    parser.add_argument('--device',     default='cuda')
    parser.add_argument('--epochs',     type=int, default=100)
    args = parser.parse_args()

    if args.mode == 'train':
        print("=== Training U-Net ===")
        model, history = train_unet(
            data_dir=args.data_dir, out_dir=args.out_dir,
            n_epochs=args.epochs, device=args.device)

    elif args.mode == 'simulate':
        print(f"=== Generating {args.n_sequences} synthetic sequences ===")
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        ds = SyntheticAAADataset(n_samples=args.n_sequences, augment=False)
        imgs  = np.stack([ds[i]['image'].numpy()[0] for i in range(len(ds))])
        masks = np.stack([ds[i]['mask'].numpy()[0]  for i in range(len(ds))])
        np.save(f'{args.out_dir}/sim_images.npy', imgs)
        np.save(f'{args.out_dir}/sim_masks.npy',  masks)
        print(f"Saved {args.n_sequences} frames to {args.out_dir}/")

    elif args.mode == 'infer':
        if args.sequence is None:
            raise ValueError("Provide --sequence path to .npy file of shape (T, H, W)")
        frames = np.load(args.sequence).astype(np.float32)
        pipeline = AAAStrainPipeline(args.weights, device=args.device)
        result = pipeline.run(frames)
        pipeline.visualise(frames[0], result, out_path='strain_output.png')
        np.save('strain_result.npy', result['strain'])

    elif args.mode == 'demo':
        print("=== Demo Mode (no GPU, no weights required) ===")
        # Create a tiny synthetic sequence
        np.random.seed(0)
        ds = SyntheticAAADataset(n_samples=10, augment=False)
        frames = np.stack([ds[i]['image'].numpy()[0] for i in range(10)])

        # Test U-Net forward pass
        model = UNet()
        x = torch.from_numpy(frames[:1, None]).float()
        with torch.no_grad():
            out = model(x)
        print(f"U-Net output shape: {out.shape}  (should be [1,1,256,256])")

        # Test optical flow
        tracker = DenseOpticalFlowTracker()
        mask = np.ones((256, 256), bool)
        flow = tracker.compute_flow(frames[0], frames[1])
        print(f"Flow field shape: {flow.shape}  (should be [256,256,2])")
        print(f"Flow magnitude max: {np.linalg.norm(flow, axis=-1).max():.3f} px")

        # Metrics demo
        pred = (out[0,0].numpy() > 0.5)
        gt   = ds[0]['mask'].numpy()[0].astype(bool)
        print(f"Demo Dice: {dice_score(pred, gt):.4f}")
        print(f"Demo IoU:  {iou_score(pred, gt):.4f}")
        print("\nDemo completed successfully. Use --mode train to train the model.")


if __name__ == '__main__':
    main()
