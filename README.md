# MFCC Temporal Wake Word Training

Train a custom wake word detector using temporal MFCC features and a GradientBoosting classifier, exported to ONNX for lightweight deployment.

## How it works

Traditional MFCC summary statistics (mean/std across all frames) lose temporal information — any two-syllable phrase looks the same. This trainer fixes that by dividing the audio into **8 time segments** and computing features per segment, preserving *when* sounds happen in the utterance.

### Feature vector (362 dimensions)

| Feature | Dimensions | Purpose |
|---|---|---|
| Per-segment MFCC mean + std | 8 x 13 x 2 = 208 | What sounds occur in each time slice |
| Per-segment MFCC deltas | 8 x 13 = 104 | Transitions between phonemes |
| Per-segment energy (RMS) | 8 | Volume envelope over time |
| Per-segment spectral centroid | 8 | Pitch movement |
| Per-segment zero crossing rate | 8 | Fricative and sibilant detection |
| Global MFCC mean + std | 13 x 2 = 26 | Overall tonal context |

### Data augmentation

Each positive sample generates 16 augmented variants:
- **Volume scaling** (6x): 0.5x to 1.6x gain
- **Speed perturbation** (4x): 0.9x to 1.1x playback rate
- **Noise injection** (3x): Gaussian noise at varying amplitudes
- **Time shift** (3x): 50/100/150ms offset

## Directory structure

```
Temporal/
  train_mfcc_temporal.py    # Training script
  positive/                 # Wake word .wav clips (16kHz mono, ~2 sec)
  negative/                 # Other speech .wav clips
  ambient_negatives/        # Real ambient audio from the deployment environment
```

## Usage

```bash
# Place your training .wav files in positive/, negative/, and ambient_negatives/
python3 train_mfcc_temporal.py
```

Or point to a different data directory:

```bash
TRAINING_DIR=/path/to/training/data python3 train_mfcc_temporal.py
```

Outputs a `.onnx` model file ready for inference with ONNX Runtime on any platform.

## Requirements

```
numpy
scipy
scikit-learn
skl2onnx
```

## Training data tips

- Record positive samples through the **same microphone** your device uses in production
- Record ambient negatives from the **actual deployment environment** — appliances, TV, conversations, pets
- More ambient negatives = fewer false triggers. 1,000+ ambient clips recommended.
- Include speech negatives with similar cadence to your wake word: greetings, names, short phrases
