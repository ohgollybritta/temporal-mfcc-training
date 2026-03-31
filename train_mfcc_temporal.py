#!/usr/bin/env python3
"""MFCC wake word with TEMPORAL features — preserves frame order."""
import numpy as np
import wave, os, glob, random
from scipy.signal import get_window
from scipy.fftpack import dct
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

BASE = os.environ.get("SAGE_TRAINING_DIR", os.path.dirname(os.path.abspath(__file__)))
random.seed(42)

def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

_FB = None
def mel_filterbank(nf, fft, sr):
    mp = np.linspace(hz_to_mel(0), hz_to_mel(sr/2), nf+2)
    hp = mel_to_hz(mp)
    b = np.floor((fft+1)*hp/sr).astype(int)
    fb = np.zeros((nf, fft//2+1))
    for i in range(nf):
        for j in range(b[i], b[i+1]): fb[i,j] = (j-b[i]) / max(b[i+1]-b[i],1)
        for j in range(b[i+1], b[i+2]): fb[i,j] = (b[i+2]-j) / max(b[i+2]-b[i+1],1)
    return fb

def extract_mfcc_frames(samples_int16, sr=16000, num_mfcc=13, num_filters=26, fft_size=512):
    global _FB
    samples = samples_int16.astype(np.float32) / 32768.0
    emph = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    fl = int(0.025*sr); fs = int(0.01*sr)
    nf = max(1, 1+(len(emph)-fl)//fs)
    padded = np.append(emph, np.zeros(nf*fs+fl-len(emph)))
    idx = (np.tile(np.arange(fl),(nf,1)) + np.tile(np.arange(0,nf*fs,fs),(fl,1)).T)
    frames = padded[idx.astype(int)] * get_window('hamming', fl)
    mag = np.absolute(np.fft.rfft(frames, fft_size))
    power = mag**2 / fft_size
    if _FB is None: _FB = mel_filterbank(num_filters, fft_size, sr)
    ms = np.dot(power, _FB.T)
    ms = np.where(ms==0, np.finfo(float).eps, ms)
    lm = np.log(ms)
    mfcc = dct(lm, type=2, axis=1, norm='ortho')[:, :num_mfcc]
    d = np.zeros_like(mfcc)
    for t in range(1, len(mfcc)-1): d[t] = (mfcc[t+1]-mfcc[t-1])/2
    d[0]=d[1]; d[-1]=d[-2]
    return mfcc, d, lm

def extract_features(samples_int16, sr=16000, n_segments=8):
    """
    Temporal MFCC features — divide frames into segments, compute stats per segment.
    This preserves WHEN things happen, not just what.
    """
    mfcc, deltas, log_mel = extract_mfcc_frames(samples_int16, sr)
    n_frames = len(mfcc)
    seg_size = max(1, n_frames // n_segments)

    features = []

    # Per-segment MFCC means and stds (8 segments × 13 coeffs × 2 stats = 208)
    for s in range(n_segments):
        start = s * seg_size
        end = min(start + seg_size, n_frames)
        seg = mfcc[start:end]
        features.extend(seg.mean(axis=0))
        features.extend(seg.std(axis=0))

    # Per-segment delta means (8 × 13 = 104) — captures transitions between phonemes
    for s in range(n_segments):
        start = s * seg_size
        end = min(start + seg_size, n_frames)
        features.extend(deltas[start:end].mean(axis=0))

    # Per-segment energy (8 values)
    samples = samples_int16.astype(np.float32) / 32768.0
    seg_len = max(1, len(samples) // n_segments)
    for s in range(n_segments):
        chunk = samples[s*seg_len:(s+1)*seg_len]
        features.append(np.sqrt(np.mean(chunk**2)))

    # Per-segment spectral centroid (8 values) — tracks pitch movement
    sc = np.sum(log_mel * np.arange(log_mel.shape[1]), axis=1) / (np.sum(log_mel, axis=1) + 1e-10)
    for s in range(n_segments):
        start = s * seg_size
        end = min(start + seg_size, n_frames)
        features.append(sc[start:end].mean())

    # Per-segment zero crossing rate (8 values) — sibilants have high ZCR
    for s in range(n_segments):
        chunk = samples[s*seg_len:(s+1)*seg_len]
        if len(chunk) > 1:
            features.append(np.sum(np.abs(np.diff(np.sign(chunk)))) / (2*len(chunk)))
        else:
            features.append(0)

    # Global stats for context (26 values)
    features.extend(mfcc.mean(axis=0))  # 13
    features.extend(mfcc.std(axis=0))   # 13

    return np.array(features, dtype=np.float32)

def load_wav(path, target=32000):
    with wave.open(path, 'rb') as wf:
        raw = wf.readframes(wf.getnframes())
        s = np.frombuffer(raw, dtype=np.int16)
    if len(s) > target: s = s[:target]
    elif len(s) < target: s = np.pad(s, (0, target-len(s)))
    return s

def augment(samples, sr=16000):
    aug = []
    for v in [0.5, 0.7, 0.85, 1.15, 1.3, 1.6]:
        aug.append(np.clip(samples*v, -32768, 32767).astype(np.int16))
    for sp in [0.9, 0.95, 1.05, 1.1]:
        idx = np.linspace(0, len(samples)-1, int(len(samples)/sp)).astype(int)
        st = samples[np.clip(idx, 0, len(samples)-1)]
        if len(st)>32000: st=st[:32000]
        else: st=np.pad(st,(0,32000-len(st)))
        aug.append(st)
    for nl in [30, 60, 100]:
        n = np.random.normal(0, nl, len(samples)).astype(np.int16)
        aug.append(np.clip(samples.astype(np.int32)+n, -32768, 32767).astype(np.int16))
    for ms in [50, 100, 150]:
        sh = int(sr*ms/1000)
        shifted = np.zeros_like(samples)
        shifted[sh:] = samples[:-sh]
        aug.append(shifted)
    return aug

# Load
print("Loading samples...")
pos_files = sorted(glob.glob(f"{BASE}/positive/*.wav"))
neg_files = sorted(glob.glob(f"{BASE}/negative/*.wav"))
amb_files = sorted(glob.glob(f"{BASE}/ambient_negatives/*.wav"))
print(f"  Pos: {len(pos_files)}  Neg: {len(neg_files)}  Amb: {len(amb_files)}")

print("\nExtracting temporal MFCC features...")
X_pos = []
for f in pos_files:
    s = load_wav(f)
    X_pos.append(extract_features(s))
    for a in augment(s):
        X_pos.append(extract_features(a))
print(f"  Positives: {len(X_pos)}")

X_neg = []
for f in neg_files:
    X_neg.append(extract_features(load_wav(f)))
for f in amb_files:
    X_neg.append(extract_features(load_wav(f)))
print(f"  Negatives: {len(X_neg)}")

X = np.array(X_pos + X_neg)
y = np.array([1]*len(X_pos) + [0]*len(X_neg))
print(f"\nTotal: {len(X)} ({len(X_pos)} pos, {len(X_neg)} neg)")
print(f"Feature dim: {X.shape[1]}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining GradientBoosting (temporal MFCC)...")
clf = GradientBoostingClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.08,
    subsample=0.8, min_samples_leaf=5, random_state=42
)
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_te)
y_proba = clf.predict_proba(X_te)

print(f"\n{'='*60}")
print(classification_report(y_te, y_pred, target_names=["neg", "pos"]))
cm = confusion_matrix(y_te, y_pred)
print(f"TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

pp = y_proba[y_te==1][:,1]
np_ = y_proba[y_te==0][:,1]
print(f"\nPos: min={pp.min():.4f} mean={pp.mean():.4f}")
print(f"Neg: max={np_.max():.4f} mean={np_.mean():.4f}")
print(f"Gap: {pp.min() - np_.max():.4f}")

# Test on ambient specifically
print(f"\nAmbient test (300 samples):")
amb_test = random.sample(amb_files, 300)
amb_sc = []
for f in amb_test:
    feat = extract_features(load_wav(f)).reshape(1,-1)
    amb_sc.append(clf.predict_proba(feat)[0][1])
amb_sc = np.array(amb_sc)
print(f"  max={amb_sc.max():.6f} mean={amb_sc.mean():.6f}")
print(f"  >0.3: {(amb_sc>0.3).sum()}  >0.5: {(amb_sc>0.5).sum()}  >0.9: {(amb_sc>0.9).sum()}")

# Test on recorded negatives (speech that isn't "hey sage")
print(f"\nSpeech negative test ({len(neg_files)} samples):")
speech_sc = []
for f in neg_files:
    feat = extract_features(load_wav(f)).reshape(1,-1)
    speech_sc.append(clf.predict_proba(feat)[0][1])
speech_sc = np.array(speech_sc)
print(f"  max={speech_sc.max():.6f} mean={speech_sc.mean():.6f}")
print(f"  >0.3: {(speech_sc>0.3).sum()}  >0.5: {(speech_sc>0.5).sum()}  >0.9: {(speech_sc>0.9).sum()}")

# Export
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(clf, initial_types=initial_type)
out = f"{BASE}/hey_sage_mfcc_v5.onnx"
with open(out, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"\nSaved: {out} ({os.path.getsize(out)/1024:.1f} KB)")
print(f"Feature dim: {X.shape[1]}")
