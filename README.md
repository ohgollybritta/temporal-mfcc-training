# MFCC Temporal Wake Word Training

Trains a custom "hey Sage" wake word detector using temporal MFCC features and a GradientBoosting classifier, exported to ONNX for Raspberry Pi deployment.

## How it works

Traditional MFCC summary statistics (mean/std across all frames) lose temporal information — "hey sage" becomes indistinguishable from any other 2-second speech clip. This trainer fixes that by dividing the audio into **8 time segments** and computing features per segment, preserving *when* sounds happen.

### Feature vector (362 dimensions)

| Feature | Dimensions | Purpose |
|---|---|---|
| Per-segment MFCC mean + std | 8 x 13 x 2 = 208 | What sounds occur in each time slice |
| Per-segment MFCC deltas | 8 x 13 = 104 | Transitions between phonemes |
| Per-segment energy (RMS) | 8 | Volume envelope over time |
| Per-segment spectral centroid | 8 | Pitch movement |
| Per-segment zero crossing rate | 8 | Sibilant detection (the "s" in sage) |
| Global MFCC mean + std | 13 x 2 = 26 | Overall tonal context |

### Data augmentation

Each positive sample generates 16 augmented variants:
- **Volume scaling** (6x): 0.5x to 1.6x gain
- **Speed perturbation** (4x): 0.9x to 1.1x playback rate
- **Noise injection** (3x): Gaussian noise at 30/60/100 amplitude
- **Time shift** (3x): 50/100/150ms offset

## Directory structure

```
Temporal/
  train_mfcc_temporal.py    # Training script
  positive/                 # "hey sage" .wav clips (16kHz mono, ~2 sec)
  negative/                 # Other speech .wav clips
  ambient_negatives/        # Real ambient audio from the deployment environment
```

## Usage

```bash
# Place training .wav files in positive/, negative/, and ambient_negatives/
python3 train_mfcc_temporal.py
```

Or point to a different data directory:

```bash
SAGE_TRAINING_DIR=/path/to/data python3 train_mfcc_temporal.py
```

Outputs `hey_sage_mfcc_v5.onnx` — copy this to your Pi at `/home/sage/hey_sage_mfcc_v5.onnx`.

## Requirements

```
numpy
scipy
scikit-learn
skl2onnx
```

## Training data tips

- Record positive samples through the **same microphone** Sage uses in production (e.g., Jabra SPEAK 510)
- Record ambient negatives from the **actual deployment room** — TV, kitchen sounds, conversations, pets
- More ambient negatives = fewer false triggers. 1,000+ ambient clips recommended.
- Include speech negatives with similar cadence: "hey Google", "hey Siri", names, short phrases
