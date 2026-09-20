# FeMMA: Federated Few-Shot Audio Deepfake Detection

FeMMA is a binary audio deepfake detector designed for the setting where different organizations hold different attack families and only a few forged examples are available for each family. The model keeps the large pretrained speech and language backbones frozen and trains only the downstream detection layers. In the federated setting, each client keeps its own audio and attack-specific few-shot samples locally, while model parameters and compact class-level feature statistics are used to transfer information across clients.

The detector combines three complementary views of an utterance:

* **Discriminative view:** a binary classifier operating directly on the learned joint embedding.
* **Attack-aware metric view:** distances to a bona-fide prototype and multiple attack-specific prototypes.
* **Semantic view:** alignment between the audio embedding and learned CLAP-based class anchors.

The final prediction is obtained by fusing the three forgery probabilities.

A useful way to read the whole pipeline is:

```text
raw waveform
    ↓
audio normalization / 16-kHz / 4-s
    ↓
frozen XLS-R representation
    +
24-D handcrafted signal descriptor
    ↓
deterministic signal description
    ↓
frozen CLAP text embedding
    ↓
2560-D multimodal feature
    ↓
global standardization
    ↓
audio ZCA whitening
    ↓
512-D audio branch + 512-D text branch
    ↓
gated low-rank bilinear fusion
    ↓
256-D joint embedding
    ↓
┌──────────────┬─────────────────┬─────────────────┐
│ Binary head  │ Attack metric   │ CLAP alignment │
└──────────────┴─────────────────┴─────────────────┘
    ↓
weighted score fusion
    ↓
final forgery probability
```

The frozen XLS-R and CLAP networks are used for representation extraction only. The federated optimizer updates the fusion, prompt, projection, and decision components.

---

# 1. End-to-End Pipeline

```text
                         INPUT AUDIO
                              │
                              ▼
                  mono / 16-kHz / 4-second crop
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
      Frozen XLS-R                    24-D signal descriptor
      Wav2Vec 2.0                          │
             │                             ▼
             │                     deterministic DSP sentence
             │                             │
             │                             ▼
             │                     Frozen CLAP text tower
             │                             │
             └──────────────┬──────────────┘
                            ▼
                 [ XLS-R || CLAP-DSP ]
                         2560-D
                            │
                            ▼
                 global standardisation
                            │
                     ┌──────┴──────┐
                     │             │
                     ▼             ▼
                audio stream    text stream
                     │             │
                     │             │
                  ZCA only         │
                     │             │
                     └──────┬──────┘
                            ▼
               gated low-rank bilinear fusion
                            │
                            ▼
                        256-D e
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
     Binary head      Prototype metric   CLAP alignment
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                     score-level fusion
                            │
                            ▼
                    p_forgery ∈ [0,1]
                            │
                            ▼
                    BONAFIDE / FORGERY
```

The main configuration exposed by the reference implementation is:

```text
SR                = 16000 Hz
MAX_SEC           = 4.0 s
W2V_NAME          = facebook/wav2vec2-xls-r-300m
CLAP_NAME         = laion/clap-htsat-unfused

W2V_DIM           = 2048
TXT_DIM           = 512
DSP_DIM           = 24

ENC_HID           = 512
EMB_DIM           = 256

N_CLIENTS         = 4
ROUNDS            = 5
LOCAL_EPOCHS      = 6
LOCAL_BS          = 128

LR                = 1e-3
WD                = 1e-4

FEDPROX_MU        = 1e-3

USE_WHITEN        = True
WHITEN_EPS        = 0.10
FUSE_GATE_INIT    = -1.0

PROMPT_MODE       = "soft"
N_CTX             = 8

W_ALIGN           = 0.3
W_PROTO           = 0.4
W_MEANSUP         = 0.6

GAME_T            = 5.0
GAME_R            = 3.0
GAME_P            = 1.0
GAME_S            = 0.0
GAME_KAPPA        = 10.0
GAME_TAU          = 0.30
W_GAME_MAX        = 0.20
GAME_WARMUP_ROUNDS= 2

FUSE_HEAD         = 0.50
FUSE_METRIC       = 0.35
FUSE_ALIGN        = 0.15

DP_SIGMA          = 0.0
DP_CLIP           = 1.0
```

The code initializes the random state with:

```text
SEED = 1234
```

and seeds Python, NumPy, PyTorch, and CUDA when available.

---

# 2. Federated Few-Shot Setting

Let one labeled utterance be

```text
(x_i, y_i, a_i)
```

where:

```text
x_i : waveform
y_i : 0 = bona fide, 1 = forgery
a_i : attack family, defined only for forged samples
```

For client `k`:

```text
D_k =
    D_k^bonafide
    ∪
    ⋃_{a ∈ A_k} D_{k,a}^forgery
```

where `A_k` is the subset of attack families owned by client `k`.

The federation is non-IID because attack families are distributed differently:

```text
Client 1 → bona fide + attacks A01, A05, A09, ...
Client 2 → bona fide + attacks A02, A06, A10, ...
Client 3 → bona fide + attacks A03, A07, A11, ...
...
```

The reason for making the federation attack-disjoint is to reproduce the harder cross-organization case:

```text
client k does NOT directly observe every attack family
```

This is where ordinary local binary training becomes problematic: a client cannot learn a decision boundary for an attack family for which it has no examples.

The few-shot budget is controlled by `N`:

```text
N = number of real forged examples retained per attack family
```

No synthetic waveform augmentation is used in the local training pool.

The code implements the partition as:

```text
1. Collect all bona-fide training items.
2. Split bona-fide items across K clients.
3. Group forged items by attack family.
4. Sort attack families.
5. Assign attack a to client:

       k = index(a) mod K

6. Shuffle that attack's samples.
7. Keep only the first N samples for its owner.
```

Thus, an attack family belongs to one client, while bona-fide data are sharded across all clients.

### Important configuration note

The pasted README text states a main configuration of:

```text
K = 4
N = 5
```

whereas the current reference script exposes:

```text
N_SHOT = 10
```

as its primary code default, with `[5, 10, 20]` used for the shot sweep.

Therefore, when reproducing the exact current script, use the value defined in `CFG.N_SHOT`. When describing the paper's reported experiment, use the experiment table/configuration actually reported in the manuscript.

---

# 3. Audio Preprocessing

For every recording:

```text
x ← load audio
x ← convert to mono
x ← resample to 16 kHz
x ← crop/pad to 4 seconds
```

The implementation uses:

```text
SR      = 16000
MAX_SEC = 4.0
L       = SR × MAX_SEC = 64000 samples
```

If the original recording is longer than `64000` samples:

```text
x = x[0 : 64000]
```

If it is shorter:

```text
x = zero_pad(x, 64000 - len(x))
```

The motivation is to make every example have the same temporal support before feature extraction, which makes the frozen speech encoder and DSP pipeline deterministic.

### XLS-R input normalization

Before sending batches to XLS-R, the code additionally normalizes every waveform:

```text
x_norm =
    (x - mean(x))
    /
    (std(x) + 1e-6)
```

The `1e-6` term prevents numerical instability for extremely low-energy signals.

This normalization is applied only for XLS-R feature extraction. It is not the same operation as the later 2560-D feature standardization.

---

# 4. Frozen XLS-R Audio Representation

The audio backbone is:

```text
facebook/wav2vec2-xls-r-300m
```

Its parameters are loaded once and frozen:

```text
for p in XLSR.parameters():
    p.requires_grad = False
```

Therefore:

```text
XLS-R weights:
    initialized from pretrained checkpoint
    never updated by FeMMA
```

For an utterance:

```text
H = [h_1, h_2, ..., h_T]
```

where each hidden state is:

```text
h_t ∈ R^1024
```

The code does not keep the full temporal sequence.

Instead:

```text
μ_H = mean_t(h_t)
σ_H = std_t(h_t)
```

and concatenates them:

```text
z_aud = [ μ_H || σ_H ]
```

giving:

```text
z_aud ∈ R^2048
```

or equivalently:

```text
1024 mean features
+
1024 standard-deviation features
=
2048 dimensions
```

### Why mean + standard deviation?

The mean captures the average learned acoustic content of the recording, while the temporal standard deviation preserves information about how much the representation varies over time.

This is preferable here to passing the full sequence because:

```text
T can be large
↓
full sequence is expensive to store
↓
temporal mean + variation gives a compact fixed-length descriptor
↓
descriptor can be cached once
```

Because XLS-R is frozen, the 2048-D representation is extracted once and cached.

---

# 5. 24-D Signal Descriptor

The second input stream is computed directly from the waveform.

The descriptor is:

```text
d_i ∈ R^24
```

and is deterministic.

The 24 dimensions are:

```text
1.  f0_mean
2.  f0_std
3.  jitter
4.  voiced_frac
5.  f0_smoothness

6.  shimmer
7.  HNR
8.  CPP

9.  spectral centroid
10. spectral bandwidth
11. spectral flatness
12. spectral roll-off

13. spectral entropy
14. spectral skewness
15. spectral kurtosis
16. spectral contrast

17. zero-crossing mean
18. zero-crossing standard deviation

19. high-frequency energy ratio
20. very-high-frequency energy ratio
21. group-delay variance

22. low-band modulation energy
23. high-band modulation energy
24. MFCC-delta temporal variance
```

The code uses:

```text
nfft = 1024
hop  = 256
eps  = 1e-8
```

for the DSP calculations.

---

## 5.1 Pitch dynamics

The fundamental frequency is estimated using YIN:

```text
f0 = librosa.yin(
    y,
    fmin=70,
    fmax=400,
    sr=16000,
    frame_length=1024,
    hop_length=256
)
```

Only finite values satisfying:

```text
70 < F0 < 400
```

are treated as voiced.

If fewer than two valid voiced values exist, the implementation substitutes:

```text
voiced = [0, 0]
```

### Mean pitch

```text
f0_mean = mean(voiced)
```

### Pitch variation

```text
f0_std = std(voiced)
```

### Jitter

The implementation defines jitter as:

```text
jitter =
    mean(|voiced[t] - voiced[t-1]|)
    /
    (mean(voiced) + eps)
```

The motivation is to measure local pitch instability rather than only average pitch.

### Voiced fraction

```text
voiced_frac =
    number_of_valid_voiced_frames
    /
    (number_of_F0_frames + eps)
```

### F0 smoothness

The second-order temporal difference is used:

```text
f0_smoothness =
    mean(|Δ²F0|)
    /
    (mean(F0) + eps)
```

This is intended to retain a contour-smoothness cue that may be informative for converted or over-smoothed speech.

---

# 6. Voice-Quality Features

## 6.1 Shimmer

Frame-level RMS energy is first computed:

```text
rms = librosa.feature.rms(
    y=y,
    frame_length=1024,
    hop_length=256
)
```

Then:

```text
shimmer =
    std(rms)
    /
    (mean(rms) + eps)
```

This captures relative amplitude fluctuation.

## 6.2 Harmonics-to-Noise Ratio

The waveform is decomposed using harmonic-percussive source separation:

```text
harm, perc = librosa.effects.hpss(y)
```

The code then uses:

```text
HNR =
    10 log10(
        (Σ harm² + eps)
        /
        (Σ perc² + eps)
    )
```

HNR is the **Harmonics-to-Noise Ratio**.

The motivation is that synthetic and converted speech can alter the balance between harmonic structure and residual/noise-like content.

## 6.3 Cepstral Peak Prominence

The STFT magnitude is:

```text
S = |STFT(y)| + eps
```

and the log-power spectrum is:

```text
logS = log(S² + eps)
```

The real cepstrum is computed by inverse FFT:

```text
cep = irfft(logS)
```

The pitch-related quefrency interval is:

```text
q_lo = int(sr / 400)
q_hi = int(sr / 70)
```

Within this range:

```text
peak = maximum cepstral value
base = mean cepstral value
```

and:

```text
CPP = mean(peak - base)
```

CPP is **Cepstral Peak Prominence**.

---

# 7. Spectral Features

Define:

```text
S = |STFT(y)| + eps
freqs = FFT frequencies
```

The code uses four standard spectral-shape measurements.

### Spectral centroid

```text
spectral_centroid =
    mean(librosa.feature.spectral_centroid(S=S))
```

### Spectral bandwidth

```text
spectral_bandwidth =
    mean(librosa.feature.spectral_bandwidth(S=S))
```

### Spectral flatness

```text
spectral_flatness =
    mean(librosa.feature.spectral_flatness(S=S))
```

### Spectral roll-off

The roll-off percentage is fixed to:

```text
roll_percent = 0.85
```

so:

```text
spectral_rolloff =
    mean(
        librosa.feature.spectral_rolloff(
            S=S,
            sr=16000,
            roll_percent=0.85
        )
    )
```

These features describe where spectral energy lies, how concentrated it is, and how rapidly it spreads toward higher frequencies.

---

# 8. Spectral Distribution Features

For every time frame, convert the magnitude spectrum to a normalized frequency distribution:

```text
P = S / (Σ_f S_f + eps)
```

Thus each frame approximately satisfies:

```text
Σ_f P_f = 1
```

## 8.1 Spectral entropy

```text
spectral_entropy =
    mean[
        -Σ_f P_f log(P_f + eps)
    ]
```

A flatter spectrum produces higher entropy, while a concentrated spectrum gives lower entropy.

## 8.2 Spectral skewness

For each frame let:

```text
μ = Σ_f P_f f
var = Σ_f P_f (f - μ)²
```

Then:

```text
spectral_skewness =
    mean[
        Σ_f P_f (f - μ)³
        /
        var^(3/2)
    ]
```

## 8.3 Spectral kurtosis

Similarly:

```text
spectral_kurtosis =
    mean[
        Σ_f P_f (f - μ)^4
        /
        var²
    ]
```

## 8.4 Spectral contrast

The code uses:

```text
spectral_contrast =
    mean(
        librosa.feature.spectral_contrast(S=S, sr=16000)
    )
```

Together, entropy, skewness, kurtosis, and contrast give a compact description of how spectral energy is distributed rather than only where its center lies.

---

# 9. Temporal, High-Frequency, and Phase Features

## 9.1 Zero-crossing dynamics

For every frame:

```text
zcr = zero_crossing_rate(
    y,
    frame_length=1024,
    hop_length=256
)
```

and:

```text
zcr_mean = mean(zcr)
zcr_std  = std(zcr)
```

These capture temporal sign-change behavior and high-frequency/noise-related variation.

## 9.2 High-frequency energy ratio

Let:

```text
tot_t = Σ_f S(f,t) + eps
```

Then:

```text
HF_ratio =
    mean[
        Σ_{f > 4000 Hz} S(f,t)
        /
        tot_t
    ]
```

## 9.3 Very-high-frequency energy ratio

```text
VHF_ratio =
    mean[
        Σ_{f > 7000 Hz} S(f,t)
        /
        tot_t
    ]
```

The motivation is to expose high-frequency attenuation or abnormal high-frequency structure.

## 9.4 Group-delay variance

The phase matrix is differentiated along frequency:

```text
gd = diff(
    angle(STFT(y)),
    axis=frequency
)
```

The wrapped phase difference is:

```text
gd_wrap =
    mod(gd + π, 2π) - π
```

and:

```text
gd_var = var(gd_wrap)
```

This introduces a phase-structure descriptor complementary to magnitude-only spectral features.

---

# 10. Temporal Modulation and MFCC Dynamics

The RMS envelope is centered:

```text
env = rms - mean(rms)
```

With:

```text
fr = sr / hop = 62.5 Hz
```

the envelope spectrum is:

```text
Menv = |FFT(env)| + eps
```

and the corresponding modulation frequencies are obtained using:

```text
mfreq = rfftfreq(
    len(env),
    d=1/fr
)
```

The normalization denominator is:

```text
denom = Σ Menv + eps
```

### Low modulation energy

```text
mod_low =
    Σ Menv[2 Hz ≤ f < 8 Hz]
    /
    denom
```

### High modulation energy

```text
mod_high =
    Σ Menv[8 Hz ≤ f < 20 Hz]
    /
    denom
```

### MFCC-delta variance

The code computes:

```text
mfcc = librosa.feature.mfcc(
    y=y,
    sr=16000,
    n_mfcc=20,
    n_fft=1024,
    hop_length=256
)
```

then:

```text
dmfcc = librosa.feature.delta(mfcc)
```

and finally:

```text
mfcc_delta_var =
    mean(
        variance(dmfcc, axis=1)
    )
```

This gives a compact descriptor of temporal evolution in cepstral coefficients.

Finally, all non-finite values are made safe:

```text
out = nan_to_num(
    out,
    nan=0,
    posinf=0,
    neginf=0
)
```

---

# 11. Signal-to-Text Conversion

The 24-D descriptor is not passed directly into a language model.

Instead, it is rendered deterministically into a textual sentence.

For example:

```text
a speech recording with mean pitch 145 hertz,
pitch variation 21 hertz, jitter 0.012,
harmonic to noise ratio 18.3 decibels,
cepstral peak prominence 2.41,
spectral flatness 0.031,
spectral entropy 4.72,
high frequency energy ratio 0.081,
group delay variance 0.44,
low band modulation 0.19
and temporal dynamics 0.72
```

The important point is that this is **template-based**, not free-form text generation.

Thus:

```text
d_i
  ↓
DSP_TO_TEXT(d_i)
  ↓
q_i
```

where `q_i` is deterministic for a given descriptor.

This gives the model a second semantic description of the recording without requiring a generative language model.

---

# 12. Frozen CLAP Text Representation

The CLAP backbone is:

```text
laion/clap-htsat-unfused
```

and is frozen:

```text
for p in CLAP.parameters():
    p.requires_grad = False
```

Therefore the base CLAP representation itself never changes.

The deterministic signal sentence is tokenized with:

```text
max_length = 64
truncation = True
padding = True
```

and passed through the frozen CLAP text tower.

The result is:

```text
z_txt ∈ R^512
```

The code normalizes the CLAP text feature:

```text
z_txt ← z_txt / ||z_txt||₂
```

and stores the 512-D result in the feature cache.

---

# 13. Cached Multimodal Feature

The two frozen representations are concatenated:

```text
z = [ z_aud || z_txt ]
```

where:

```text
z_aud ∈ R^2048
z_txt ∈ R^512
```

so:

```text
z ∈ R^2560
```

The cache therefore stores:

```text
cache[item] =
    [
        z      # float32, 2560-D
        d      # float32, 24-D DSP descriptor
    ]
```

The cache is created once because:

```text
XLS-R is frozen
CLAP is frozen
DSP extraction is deterministic
DSP-to-text is deterministic
```

This removes the expensive backbone computation from every federated training round.

---

# 14. Global Feature Standardisation

The cached 2560-D feature has heterogeneous scales across dimensions.

Therefore, the implementation computes global feature-wise statistics:

```text
μ = mean(z)
σ = std(z) + 1e-6
```

and standardizes every feature as:

```text
z_s =
    (z - μ)
    /
    σ
```

Here:

```text
μ ∈ R^2560
σ ∈ R^2560
```

and each dimension is standardized independently.

The added `1e-6` avoids division by zero.

### Exact code lifecycle

The current reference script computes:

```text
Zall = stack(all cached z)
μ    = mean(Zall, axis=0)
σ    = std(Zall, axis=0) + 1e-6
```

before constructing the federated training partitions.

Thus:

```text
μ and σ are fixed for the entire run
```

They are not learned by gradient descent.

They are stored inside the model as buffers:

```text
gmu
gsd
```

and remain unchanged during AdamW updates.

---

# 15. Audio/Text Split

After standardization:

```text
z_aud^s = z_s[0 : 2048]
z_txt^s = z_s[2048 : 2560]
```

Therefore:

```text
z_aud^s ∈ R^2048
z_txt^s ∈ R^512
```

Only the audio portion is ZCA-whitened.

The text stream remains standardized but not whitened.

---

# 16. ZCA Whitening

The motivation for ZCA is that the XLS-R dimensions are highly correlated.

For the standardized audio matrix `A`:

```text
A ∈ R^(N × 2048)
```

the code computes:

```text
C = (Aᵀ A) / N
```

and explicitly symmetrizes it:

```text
C = 0.5 (C + Cᵀ)
```

Then:

```text
C = V Λ Vᵀ
```

where:

```text
V : eigenvector matrix
Λ : diagonal eigenvalue matrix
```

Negative numerical eigenvalues are clamped:

```text
λ_j ← max(λ_j, 0)
```

The ZCA matrix is:

```text
W_ZCA =
    V
    diag(
        1 / sqrt(λ_j + ε_w)
    )
    Vᵀ
```

with:

```text
ε_w = 0.10
```

The implementation therefore uses:

```text
W_ZCA =
    V (Λ + 0.10 I)^(-1/2) Vᵀ
```

and:

```text
z_aud^w = W_ZCA z_aud^s
```

The whitening matrix is computed once and then stored as a non-trainable buffer:

```text
Wa = W_ZCA
```

It is **not** updated by AdamW.

When whitening is disabled:

```text
USE_WHITEN = False
```

the code uses:

```text
Wa = I
```

instead.

### Why ZCA?

The goal is:

```text
correlated audio features
        ↓
decorrelated audio features
        ↓
more stable downstream fusion
```

ZCA is used instead of an arbitrary rotation because it maintains the original coordinate orientation as much as possible.

---

# 17. Cross-Modal Fusion Encoder

The trainable encoder has:

```text
ENC_HID = 512
EMB_DIM = 256
```

The audio and text branches are projected separately.

---

## 17.1 Audio branch

```text
a =
    Linear_2048→512(z_aud^w)
    ↓
    LayerNorm
    ↓
    GELU
    ↓
    Dropout(p=0.3)
```

Mathematically:

```text
h_aud =
    Dropout(
        GELU(
            LN(
                W_a z_aud^w + b_a
            )
        )
    )
```

The trainable parameters are:

```text
W_a
b_a
LayerNorm parameters
```

and are initialized by PyTorch's standard module initialization and then updated by AdamW.

---

## 17.2 Text branch

Similarly:

```text
h_txt =
    Dropout(
        GELU(
            LN(
                W_t z_txt^s + b_t
            )
        )
    )
```

with:

```text
W_t : 512 → 512
b_t : 512
```

and dropout probability:

```text
p_dropout = 0.3
```

---

# 18. Gated Low-Rank Bilinear Fusion

The two 512-D branches are projected with:

```text
U_a : 512 → 512
U_t : 512 → 512
```

and no bias.

The multiplicative cross-modal interaction is:

```text
u =
    tanh(U_a h_aud)
    ⊙
    tanh(U_t h_txt)
```

where:

```text
⊙ = element-wise multiplication
```

The motivation is important.

A simple concatenation says:

```text
audio feature + text feature
```

whereas the multiplicative term explicitly asks:

```text
which audio dimensions become important
when a particular semantic/signal dimension is also active?
```

---

# 19. Learnable Fusion Gate

The code uses a learnable 512-D gate parameter:

```text
gate ∈ R^512
```

initialized as:

```text
gate_j = -1.0
```

for every dimension.

The actual gate value is:

```text
g = sigmoid(gate)
```

so initially:

```text
sigmoid(-1) ≈ 0.269
```

Therefore, the multiplicative interaction starts relatively weak.

The fused representation is:

```text
h =
    h_aud
    +
    sigmoid(gate) ⊙ u
```

This gives an audio-dominant initialization.

The motivation is to prevent an unstable early-stage semantic interaction from overwhelming the already strong speech representation.

Unlike the ZCA matrix, the gate **is trainable**.

During every AdamW update:

```text
gate
    ↓
gradient
    ↓
AdamW
    ↓
updated gate
```

so the model learns how strongly each hidden dimension should use cross-modal interaction.

---

# 20. Projection to the 256-D Joint Embedding

After gated fusion:

```text
h
  ↓
LayerNorm
  ↓
GELU
  ↓
Linear 512 → 256
  ↓
LayerNorm
  ↓
GELU
```

giving:

```text
e ∈ R^256
```

This is the central representation used by all three decision branches.

The final embedding projection is trainable and is updated by all losses.

---

# 21. Signal-Conditioned CLAP Class Anchors

FeMMA also learns two semantic anchors:

```text
BONAFIDE
FORGERY
```

The implementation starts from fixed human-written class descriptions.

The frozen CLAP text encoder turns these into:

```text
base_anchor ∈ R^(2×512)
```

These base embeddings are normalized and stored as a frozen buffer.

---

## 21.1 Learnable soft prompt

The default mode is:

```text
PROMPT_MODE = "soft"
N_CTX = 8
```

so each class receives:

```text
8 learnable context tokens
```

If the CLAP text hidden size is `H_txt`, each class context is:

```text
ctx_c ∈ R^(8 × H_txt)
```

The initialization is:

```text
ctx_c ~ Normal(0, 0.02²)
```

These parameters are trainable.

Therefore, unlike the frozen CLAP weights:

```text
soft prompt
    ↓
updated by AdamW
    ↓
aggregated by FedAvg
```

---

# 22. DSP-Conditioned Prompt Network

The descriptor:

```text
d ∈ R^24
```

is passed through a shared meta-network:

```text
24
 ↓
Linear(24 → 128)
 ↓
ReLU
 ↓
128
```

giving:

```text
r = ReLU(W_m d + b_m)
```

The code then uses:

```text
meta_soft:
    128 → text_hidden
```

for soft-prompt conditioning.

Thus:

```text
Δctx =
    meta_soft(
        meta_trunk(d)
    )
```

and the context is:

```text
ctx_conditioned =
    ctx_class
    +
    Δctx
```

The same 512-dimensional offset is broadcast across the eight context tokens.

This gives:

```text
fixed class semantics
+
learned prompt
+
signal-dependent offset
```

rather than a completely fixed text representation.

All parameters of the meta-network are trainable and FedAvg-aggregated.

---

# 23. Exact DSP Conditioning Used During Local Training

The code does not condition the prompt on every individual utterance separately.

For a client, it computes two local DSP averages:

```text
dsp_bar[0] =
    mean DSP descriptor over local bona-fide items

dsp_bar[1] =
    mean DSP descriptor over local forged items
```

so:

```text
dsp_bar ∈ R^(2×24)
```

Then, for the two semantic classes:

```text
BONAFIDE → dsp_bar[0]
FORGERY  → dsp_bar[1]
```

are used to generate the semantic anchors.

This means the local alignment branch is conditioned by the **client's class-level acoustic statistics**, not a separately generated text prompt for every sample.

---

# 24. Prompt Fallback Behavior

The default mode is:

```text
PROMPT_MODE = "soft"
```

Before training, the implementation probes whether the installed CLAP version supports the required soft-prompt input.

If the probe succeeds:

```text
soft prompt mode
```

is used.

If it fails:

```text
adapter mode
```

is used automatically.

In adapter mode:

```text
ctx_c ∈ R^512
```

is initialized to zeros.

The anchor is formed from:

```text
base anchor
+
0.1 × adapter(base anchor)
+
0.1 × context
```

and, when DSP conditioning is available:

```text
+
0.1 × meta_adapter(meta_trunk(dsp))
```

The adapter itself is:

```text
Linear(512 → 512)
↓
Tanh
```

This fallback is an implementation safeguard and does not modify the frozen CLAP backbone.

---

# 25. CLAP Anchor Projection into the Detector Space

The CLAP anchor is originally:

```text
t_c ∈ R^512
```

but the detector embedding is:

```text
e ∈ R^256
```

Therefore, a trainable projection is used:

```text
text_proj :
    512 → 256
```

followed by L2 normalization:

```text
t_hat_c =
    text_proj(t_c)
    /
    ||text_proj(t_c)||₂
```

The projection layer is initialized by the default PyTorch linear initializer and updated by AdamW.

---

# 26. Learnable Alignment Temperature

The code contains one trainable scalar:

```text
logit_scale
```

initialized to:

```text
2.3
```

The actual scale used in the logits is:

```text
τ =
    exp(
        clamp(logit_scale, max=4.6)
    )
```

The initial value is therefore approximately:

```text
exp(2.3) ≈ 9.97
```

The clamp prevents the scale from exploding.

Because `logit_scale` is trainable:

```text
logit_scale
    ↓
gradient
    ↓
AdamW
    ↓
updated value
```

---

# 27. Federated Statistics Bank

The major problem in attack-disjoint federation is:

```text
Client 1 does not see A17
Client 2 does not see A03
Client 3 does not see A14
...
```

A client therefore cannot directly construct every attack prototype.

FeMMA uses a compact statistics bank instead of transferring waveforms.

For every class `c`, including:

```text
bonafide
A01
A02
...
A19
```

the implementation stores:

```text
S_c
SS_c
N_c
```

where:

```text
S_c  = Σ_i z_i^s
SS_c = Σ_i (z_i^s ⊙ z_i^s)
N_c  = number of class-c examples
```

The current implementation forms these values after the client few-shot partition.

They are aggregated into a `FeatBank`.

---

# 28. Diagonal-Gaussian Statistics

For a class with:

```text
N_c ≥ 1
```

the bank computes:

```text
μ_c =
    S_c / N_c
```

and:

```text
var_c =
    max(
        SS_c / N_c - μ_c²,
        1e-3
    )
```

The minimum variance parameter is:

```text
min_var = 1e-3
```

This avoids negative or numerically tiny variances.

The important point is:

```text
variance is stored in the bank
```

but the current detector training path primarily uses the class **means**.

No raw waveform is placed inside the statistics bank.

---

# 29. Global Mean Supervision

The bank means are converted into tensors:

```text
mf =
[
    μ_bonafide,
    μ_A01,
    μ_A02,
    ...
]
```

with corresponding binary labels:

```text
ml =
[
    0,
    1,
    1,
    ...
]
```

Each mean is passed through the **current trainable detector**:

```text
e_c = g_θ(μ_c)
```

The reason for this step is subtle but important.

Suppose:

```text
Client 1:
    does not own A17
```

Client 1 still receives:

```text
μ_A17
```

through the global statistics bank.

It can therefore update its own detector using A17 without receiving the underlying A17 waveform.

This transfers class-level information across the federation without directly sharing examples.

---

# 30. Attack-Aware Prototypes

For every class mean:

```text
μ_c
```

the current encoder generates:

```text
e_c = g_θ(μ_c)
```

and normalizes it:

```text
p_c =
    e_c / ||e_c||₂
```

giving:

```text
p_0      = bona-fide prototype
p_A01    = attack A01 prototype
p_A02    = attack A02 prototype
...
```

The critical design choice is:

```text
FORGERY ≠ one single global centroid
```

Instead:

```text
FORGERY
    ↓
{A01, A02, ..., A19}
    ↓
multiple attack-specific geometric anchors
```

This makes it possible for an utterance to be close to one attack family even when it is far from another.

---

# 31. Local Prototype Override

The code also computes:

```text
extra[c]
```

for class means belonging to the current client.

For each client:

```text
extra[c] =
    local standardized class mean
```

if that client actually owns examples of class `c`.

The prototype construction then starts from the federated bank and overwrites local classes:

```text
means = global bank means

for c in extra:
    means[c] = local class mean
```

Therefore:

```text
locally observed class
    → use local mean

locally unseen class
    → use federated bank mean
```

This gives the current client a more specific local prototype for its own attack family while retaining global prototypes for attacks it does not own.

The prototype embeddings are recomputed from the **current model** before each local epoch.

Thus:

```text
prototype parameters themselves are not learned as independent vectors
```

Instead:

```text
bank mean
    ↓
current encoder
    ↓
prototype
```

This causes the prototype geometry to change whenever the encoder changes.

---

# 32. Three Detection Branches

After producing:

```text
e ∈ R^256
```

FeMMA evaluates three complementary views.

---

# 33. Branch 1 — Discriminative Binary Head

The primary head is:

```text
Linear(256 → 1)
```

with:

```text
l_head = w_hᵀ e + b_h
```

and:

```text
p_head = sigmoid(l_head)
```

where:

```text
p_head ∈ [0,1]
```

is the direct forgery probability.

The code implements the primary loss as:

```text
L_head =
    BCEWithLogits(
        l_head,
        y
    )
```

equivalently:

```text
L_head =
    - y log(p_head)
    - (1-y) log(1-p_head)
```

The discriminative head is not multiplied by an explicit loss weight:

```text
weight(L_head) = 1.0
```

so it acts as the primary objective.

---

# 34. Branch 2 — Attack-Aware Prototype Metric

First normalize the utterance embedding:

```text
e_hat =
    e / ||e||₂
```

The bona-fide distance is:

```text
d_0 =
    ||e_hat - p_0||₂²
```

For every attack family:

```text
d_a =
    ||e_hat - p_a||₂²
```

The code uses a differentiable soft minimum across attacks:

```text
d_1 =
    -log(
        Σ_a exp(-d_a)
    )
```

This behaves approximately like the distance to the nearest attack prototype while remaining differentiable.

The metric logit uses a fixed scale:

```text
γ = 4.0
```

giving:

```text
l_metric =
    4.0 (d_0 - d_1)
```

and:

```text
p_metric =
    sigmoid(l_metric)
```

The intuition is:

```text
far from bona fide
+
close to one or more attack families
        ↓
large positive metric logit
        ↓
high forgery probability
```

The real-mini-batch metric loss is:

```text
L_proto =
    BCEWithLogits(
        l_metric,
        y
    )
```

with:

```text
W_PROTO = 0.4
```

in the total objective.

---

# 35. Branch 3 — CLAP Semantic Alignment

The two projected class anchors are:

```text
t_bona
t_forg
```

and the utterance embedding is:

```text
e_hat
```

The semantic logits are:

```text
s_c =
    τ e_hatᵀ t_c
```

where:

```text
τ =
    exp(clamp(logit_scale, max=4.6))
```

The two-class cross-entropy is:

```text
L_align =
    CrossEntropy(
        [s_bona, s_forg],
        y
    )
```

and:

```text
p_align =
    softmax([s_bona, s_forg])[FORGERY]
```

This branch asks:

```text
Does the learned embedding look semantically more consistent
with the bona-fide anchor or the forgery anchor?
```

The local loss contribution is weighted by:

```text
W_ALIGN = 0.3
```

---

# 36. Exact Role of the DSP Stream in the Alignment Branch

There are two separate DSP-related roles.

### Role 1 — Cached multimodal input

Every utterance gets:

```text
DSP descriptor
    ↓
DSP sentence
    ↓
CLAP text embedding
    ↓
512-D cached text stream
```

This affects the multimodal embedding directly.

### Role 2 — Prompt conditioning

During local alignment training:

```text
client DSP mean
    ↓
meta network
    ↓
prompt offset
    ↓
class anchor
```

The current implementation therefore has:

```text
speech representation
+
DSP-derived CLAP feature
+
DSP-informed class prompt
```

rather than relying only on one modality.

At inference, however, the current code calls:

```text
model.text_anchors(None)
```

so no new utterance-specific DSP vector is passed into the class-anchor prompt at test time. The learned class prompt itself is used.

---

# 37. Global Mean Supervision Loss

The global class means are passed through the current detector:

```text
e_c = g_θ(μ_c)
```

and evaluated using the same binary head:

```text
L_mean_head =
    BCEWithLogits(
        head_logit(e_c),
        y_c
    )
```

and prototype metric:

```text
L_mean_metric =
    BCEWithLogits(
        metric_logit(e_c),
        y_c
    )
```

Therefore:

```text
L_mean =
    L_mean_head
    +
    L_mean_metric
```

and the total weight is:

```text
W_MEANSUP = 0.6
```

This is **not** synthetic audio augmentation.

It is direct supervision from class-level feature statistics.

---

# 38. Why Mean Supervision Matters in the Non-IID Case

Without mean supervision:

```text
Client 1
  ↓
only local attack families
  ↓
local gradient
  ↓
global model may forget unseen attacks
```

With mean supervision:

```text
Client 1
  ↓
local real data
+
global attack means
  ↓
gradient contains information about unseen attacks
  ↓
better global attack coverage
```

The mean supervision mechanism therefore acts as a bridge between:

```text
local few-shot observations
```

and:

```text
global attack-family coverage
```

---

# 39. Game-Theoretic Alignment

Two branches may not behave equally well on every attack.

Let:

```text
H = discriminative head
A = CLAP alignment branch
```

For every attack `a`, compute the head and alignment probabilities on the broadcast attack mean:

```text
p_H(a)
p_A(a)
```

and also on the bona-fide mean:

```text
p_H(0)
p_A(0)
```

The separation margins are:

```text
m_H(a) =
    p_H(a) - p_H(0)

m_A(a) =
    p_A(a) - p_A(0)
```

These are not simply class probabilities.

They measure:

```text
how much better the branch separates attack a
from bona fide
```

---

# 40. Soft Cooperation Scores

The head cooperation score is:

```text
q_a =
    sigmoid(
        κ (m_H(a) - τ_g)
    )
```

The alignment cooperation score is:

```text
p_a =
    sigmoid(
        κ (m_A(a) - τ_g)
    )
```

The reference values are:

```text
κ   = 10.0
τ_g = 0.30
```

Interpretation:

```text
m ≫ τ_g
    ↓
cooperation score ≈ 1

m ≪ τ_g
    ↓
cooperation score ≈ 0
```

The large `κ=10` makes this transition relatively sharp.

Both `κ` and `τ_g` are fixed configuration values, not learned.

---

# 41. Soft Prisoner's-Dilemma Payoff

The code uses:

```text
T = 5.0
R = 3.0
P = 1.0
S = 0.0
```

with:

```text
T > R > P > S
```

The expected payoff for the alignment player is:

```text
U_a =
      p_a q_a R
    + p_a (1-q_a) S
    + (1-p_a) q_a T
    + (1-p_a)(1-q_a) P
```

Substituting the reference values:

```text
U_a =
      3 p_a q_a
    + 0 p_a(1-q_a)
    + 5(1-p_a)q_a
    + 1(1-p_a)(1-q_a)
```

The purpose is not to model literal human strategic behavior.

It is a differentiable mechanism for deciding:

```text
where does the alignment branch need to work harder?
```

---

# 42. Difficulty Weighting

The key weighting term is:

```text
w_a =
    stop_gradient(
        1 - q_a
    )
```

So:

```text
head is strong
q_a ≈ 1
    ↓
w_a ≈ 0

head is weak
q_a ≈ 0
    ↓
w_a ≈ 1
```

This means the alignment branch receives more game-theoretic pressure precisely when the primary discriminative branch has weak attack separation.

The game loss is:

```text
L_game =
    mean_a [
        w_a (-U_a)
    ]
```

The `stop_gradient` operator is important.

Without it, the model could potentially reduce the weighting mechanism itself rather than learning the intended branch correction.

---

# 43. Game-Term Warm-Up

The game term is not introduced at full strength immediately.

The function is:

```text
if GAME_WARMUP_ROUNDS <= 0:
    λ_game = W_GAME_MAX

else:
    frac =
        min(
            1,
            (round_idx + 1)
            /
            GAME_WARMUP_ROUNDS
        )

    λ_game =
        frac × W_GAME_MAX
```

With:

```text
GAME_WARMUP_ROUNDS = 2
W_GAME_MAX          = 0.20
```

the effective values are:

```text
round 1:
    λ_game = 0.10

round 2:
    λ_game = 0.20

round 3 onward:
    λ_game = 0.20
```

This avoids applying a strong game-theoretic signal while the initial decision branches are still poorly calibrated.

---

# 44. Important Game-Loss Detail

For the real mini-batch alignment loss, the code uses:

```text
text_anchors(dsp_bar)
```

with local DSP conditioning.

However, the game loss uses:

```text
text_anchors(None)
```

Therefore the game mechanism is evaluated using the learned global class anchors without passing the local DSP averages into the anchor generator.

This keeps the game signal focused on branch-level attack separation rather than making the game depend on the current client's local DSP statistics.

---

# 45. Complete Local Objective

For a real local batch:

```text
L_real =
      L_head
    + W_ALIGN L_align
    + W_PROTO L_proto
```

where:

```text
W_ALIGN = 0.3
W_PROTO = 0.4
```

Statistics-based supervision adds:

```text
+ W_MEANSUP L_mean
```

with:

```text
W_MEANSUP = 0.6
```

The game term adds:

```text
+ λ_game L_game
```

where `λ_game` is the warm-up value.

Finally, FedProx adds:

```text
L_FedProx =
    (μ_FedProx / 2)
    ||θ_local - θ_global||²
```

with:

```text
μ_FedProx = 1e-3
```

Therefore:

```text
L_total =
      L_head
    + 0.3 L_align
    + 0.4 L_proto
    + 0.6 L_mean
    + λ_game L_game
    + L_FedProx
```

where:

```text
λ_game =
    0.10 in round 1
    0.20 from round 2 onward
```

under the reference warm-up configuration.

---

# 46. What Exactly Is Updated by Gradient Descent?

The following components are trainable:

```text
CrossModalFusionEncoder
    ├── audio linear projection
    ├── audio LayerNorm
    ├── text linear projection
    ├── text LayerNorm
    ├── U_a
    ├── U_t
    ├── gate
    └── output projection

Binary head
    └── Linear(256 → 1)

CLAP anchor projection
    └── Linear(512 → 256)

CLAP logit scale
    └── scalar

Soft prompt parameters
    └── 8 context tokens per class

Prompt meta-network
    ├── 24 → 128
    ├── ReLU
    └── 128 → text hidden dimension

Adapter fallback parameters
    └── only used if soft-prompt mode is unavailable
```

The following are fixed:

```text
XLS-R weights
CLAP weights
tokenizer
μ
σ
W_ZCA
DSP extraction rules
DSP sentence template
attack-family text templates
metric scale γ = 4.0
game hyperparameters
score-fusion weights
```

---

# 47. Trainable Parameter Initialization

At model construction:

### Frozen backbones

```text
XLS-R:
    pretrained checkpoint

CLAP:
    pretrained checkpoint
```

No random initialization occurs for these modules.

### Linear/LayerNorm modules

PyTorch's default `nn.Linear` and `nn.LayerNorm` initializers are used for:

```text
audio encoder
text encoder
U_a
U_t
output projection
binary head
text projection
meta-network
adapter
```

These parameters are then updated by AdamW.

### Fusion gate

Explicitly initialized as:

```text
gate = -1.0
```

for all 512 dimensions.

### Soft prompts

Explicitly initialized as:

```text
Normal(mean=0, std=0.02)
```

for each context token.

### Alignment log scale

Explicitly initialized:

```text
logit_scale = 2.3
```

### Statistics

```text
μ  = computed from cached features
σ  = computed from cached features
W_ZCA = computed from standardized audio features
```

These are not gradient parameters.

---

# 48. Local Optimizer

Every client-local training call creates a fresh optimizer:

```text
AdamW(
    trainable_parameters,
    lr=1e-3,
    weight_decay=1e-4
)
```

Therefore:

```text
LR = 1e-3
WD = 1e-4
```

There is no learning-rate scheduler in the reference local training loop.

AdamW moments are local to that client-training invocation.

At each local optimization step:

```text
1. opt.zero_grad()
2. forward pass
3. compute total loss
4. backward()
5. gradient clipping
6. optimizer.step()
```

---

# 49. Exact Local Batch Construction

The code uses:

```text
LOCAL_BS = 128
```

and creates:

```text
half = max(1, LOCAL_BS // 2)
      = 64
```

samples per class.

So one local batch is constructed as:

```text
64 bona fide
+
64 forged
=
128 examples
```

The implementation samples separately from:

```text
idx0 = bona fide indices
idx1 = forgery indices
```

and concatenates the two groups.

If one class contains fewer than 64 samples:

```text
replace=True
```

is used for that class.

This keeps local batches balanced even when the few-shot data are small.

---

# 50. Local Training Loop

For every local client:

```text
for epoch = 1,...,6
```

and for each epoch:

```text
1. Build current prototypes.
2. Sample balanced local mini-batch.
3. Embed batch.
4. Compute binary head loss.
5. Compute CLAP alignment loss.
6. Compute prototype metric loss.
7. Compute mean supervision.
8. Compute game loss.
9. Compute FedProx.
10. Sum losses.
11. Backpropagate.
12. Clip gradients.
13. Update with AdamW.
```

The number of local epochs is:

```text
LOCAL_EPOCHS = 6
```

---

# 51. Gradient Clipping

After backpropagation:

```text
clip_grad_norm_(
    params,
    max_norm=5.0
)
```

is applied.

Thus the exact code value is:

```text
gradient_clip_max_norm = 5.0
```

The clipping happens immediately before:

```text
opt.step()
```

and therefore limits the local parameter update caused by unusually large gradients.

---

# 52. FedProx Update

At the beginning of every federated round, the server model is saved:

```text
θ_global
```

Each client starts from a copy:

```text
θ_k ← θ_global
```

During local optimization:

```text
L_FedProx =
    0.5 × 1e-3 ×
    ||θ_k - θ_global||²
```

is added.

Therefore, local training is encouraged to remain close to the received global model.

This is particularly useful when different clients contain different attack families.

---

# 53. Client–Server Interaction

A communication round is:

```text
SERVER
  │
  │ broadcast current global model
  ▼
CLIENT 1
CLIENT 2
...
CLIENT K
  │
  │ each performs local training
  ▼
updated client models
  │
  │ upload
  ▼
SERVER
  │
  │ sample-weighted FedAvg
  ▼
new global model
```

Unlike centralized training:

```text
server never needs another client's waveform
```

during the local optimization process.

---

# 54. Client Data That Remains Local

Each client keeps:

```text
raw waveform
file path
individual utterance
individual DSP descriptor
individual cached multimodal feature
local few-shot attack examples
```

inside its local training process.

The intended conceptual privacy boundary is:

```text
RAW AUDIO                 → local
INDIVIDUAL UTTERANCE      → local
LOCAL FEW-SHOT SAMPLES    → local
MODEL UPDATE              → server
AGGREGATE CLASS STATS     → server
GLOBAL MODEL              → clients
```

---

# 55. Current Reference Script: Statistics Construction

The implementation first constructs a statistics bank from the client partitions:

```text
for each client:
    compute local class statistics
        S_c
        SS_c
        N_c

aggregate them into FeatBank
```

The class statistics are based on standardized 2560-D features.

This creates:

```text
global bank:
    μ_bonafide
    μ_A01
    ...
    μ_A19
```

The bank is constructed before the federated communication loop.

In the current script, the bank itself is not re-created from uploaded statistics after every round. Instead:

```text
bank = build_bank(...)
```

is created once before:

```text
for r in range(ROUNDS):
```

The prototype embeddings are still recomputed from the bank means with the **current encoder parameters**, so the geometry changes as the model changes.

This distinction is important:

```text
BANK MEANS:
    fixed during the current run

ENCODER:
    updated every local step

PROTOTYPES:
    recomputed from fixed means using the current encoder
```

---

# 56. Server Aggregation — FedAvg

After all valid clients finish local training:

```text
w_k =
    |D_k|
    /
    Σ_j |D_j|
```

and:

```text
θ^(r+1) =
    Σ_k w_k θ_k
```

Thus clients with more local training examples receive larger aggregation weights.

The aggregation is parameter-wise.

---

# 57. Optional Differential-Privacy Upload Processing

The configuration contains:

```text
DP_SIGMA = 0.0
DP_CLIP  = 1.0
```

When:

```text
DP_SIGMA = 0
```

the update is returned unchanged.

When:

```text
DP_SIGMA > 0
```

the code applies, for each uploaded parameter tensor:

```text
v ← v × min(
          1,
          DP_CLIP / (||v||₂ + 1e-12)
      )
```

followed by:

```text
v_noisy =
    v
    +
    Normal(
        0,
        DP_SIGMA × DP_CLIP
    )
```

The default experimental run therefore uses:

```text
DP_SIGMA = 0
```

and does not perturb the upload.

---

# 58. Complete Training Algorithm

```text
INPUT
    K               number of clients
    N               few-shot forged samples per attack owner
    R               number of federated rounds
    E               local epochs
    D               available training items
    XLS-R           frozen speech encoder
    CLAP            frozen text encoder

FIXED CONFIGURATION
    SR = 16000
    MAX_SEC = 4.0
    W2V_DIM = 2048
    TXT_DIM = 512
    DSP_DIM = 24
    ENC_HID = 512
    EMB_DIM = 256

    USE_WHITEN = True
    WHITEN_EPS = 0.10
    FUSE_GATE_INIT = -1.0

    PROMPT_MODE = "soft"
    N_CTX = 8

    W_ALIGN = 0.3
    W_PROTO = 0.4
    W_MEANSUP = 0.6

    FEDPROX_MU = 1e-3

    GAME_T = 5.0
    GAME_R = 3.0
    GAME_P = 1.0
    GAME_S = 0.0
    GAME_KAPPA = 10.0
    GAME_TAU = 0.30
    W_GAME_MAX = 0.20
    GAME_WARMUP_ROUNDS = 2

    FUSE_HEAD = 0.50
    FUSE_METRIC = 0.35
    FUSE_ALIGN = 0.15

    LR = 1e-3
    WD = 1e-4
    GRAD_CLIP = 5.0


PREPROCESSING
    For every audio file x:

        x ← load(x)

        if multichannel:
            x ← mean_over_channels(x)

        if sampling_rate != 16000:
            x ← resample(x, 16000)

        if len(x) >= 64000:
            x ← x[0:64000]
        else:
            x ← zero_pad(x, 64000-len(x))

        # XLS-R extraction normalization
        x_norm ←
            (x - mean(x))
            /
            (std(x) + 1e-6)


FROZEN AUDIO FEATURE
    H ← XLS-R(x_norm)

    z_aud ←
        [
            mean(H, time),
            std(H, time)
        ]

    # z_aud is 2048-D


DSP FEATURE
    d ← DSP_DESCRIPTOR(x)

    # d is 24-D

    q ← DSP_TO_TEXT(d)

    z_txt ← CLAP_TEXT(q)

    # z_txt is 512-D

    z ← [z_aud || z_txt]

    # z is 2560-D

    Cache:
        z
        d


GLOBAL PREPROCESSING STATISTICS
    Zall ← stack(all cached z)

    μ ← mean(Zall, axis=0)

    σ ← std(Zall, axis=0) + 1e-6

    A ←
        (Zall[:, 0:2048] - μ[0:2048])
        /
        σ[0:2048]

    if USE_WHITEN:
        C ← (AᵀA) / max(1, number_of_rows(A))
        C ← 0.5(C + Cᵀ)

        [V, Λ] ← eig(C)

        Λ ← max(Λ, 0)

        W_ZCA ←
            V(
                Λ + 0.10I
            )^(-1/2)Vᵀ
    else:
        W_ZCA ← I


DATA SPLIT
    Build cached dataset.

    For every class / attack:
        shuffle examples

        n_train ← floor(0.6 × class_size)

        first n_train  → training pool
        remaining      → test pool

    Shuffle train pool.
    Shuffle test pool.


FEDERATED FEW-SHOT PARTITION
    Separate bona-fide items.

    Split bona-fide examples across K clients.

    Group spoof samples by attack family.

    For every attack a:
        owner(a) ← index(a) mod K
        shuffle attack examples
        retain first N examples
        assign retained examples to owner(a)

    No synthetic waveform examples are created.


STATISTICS BANK
    For each client k:

        FOR each class c:

            z_s ←
                (z - μ) / σ

            S_{k,c}   ← Σ z_s
            SS_{k,c}  ← Σ (z_s ⊙ z_s)
            N_{k,c}   ← number of samples

    Aggregate:

        S_c  ← Σ_k S_{k,c}
        SS_c ← Σ_k SS_{k,c}
        N_c  ← Σ_k N_{k,c}

    Compute:

        μ_c = S_c / N_c

        var_c =
            max(
                SS_c / N_c - μ_c²,
                1e-3
            )

    Store:
        μ_c
        var_c
        N_c


MODEL INITIALIZATION
    Initialize trainable model θ_0.

    Audio encoder:
        Linear(2048 → 512)
        LayerNorm
        GELU
        Dropout(0.3)

    Text encoder:
        Linear(512 → 512)
        LayerNorm
        GELU
        Dropout(0.3)

    U_a:
        Linear(512 → 512, bias=False)

    U_t:
        Linear(512 → 512, bias=False)

    gate:
        initialize every value to -1.0

    output block:
        LayerNorm
        GELU
        Linear(512 → 256)
        LayerNorm
        GELU

    binary head:
        Linear(256 → 1)

    text projection:
        Linear(512 → 256)

    logit_scale:
        initialize to 2.3

    class soft prompts:
        initialize N(0, 0.02²)

    DSP meta network:
        Linear(24 → 128)
        ReLU
        Linear(128 → CLAP-text-hidden)

    Freeze:
        XLS-R
        CLAP
        tokenizer
        μ
        σ
        W_ZCA


FOR round r = 0,...,R-1:

    SERVER
        Keep current global model θ_r.


    FOR each client k:

        local_model ← deepcopy(global_model)

        Create fresh AdamW optimizer:

            lr = 1e-3
            weight_decay = 1e-4


        Compute local class DSP means:

            dsp_bar_bona
            dsp_bar_forg

        w_game ←

            if GAME_WARMUP_ROUNDS <= 0:
                0.20

            else:
                min(
                    1,
                    (r+1)/2
                ) × 0.20


        FOR epoch e = 1,...,6:

            Build current prototypes:

                global bank means
                    +
                client's local mean overrides

            Embed each prototype with current local_model.

            Normalize prototypes.

            Construct:

                bona-fide prototype p_0
                attack prototype stack {p_a}


            Determine:

                half = 128/2 = 64


            REPEAT local steps:

                Sample 64 bona-fide examples.

                Sample 64 forged examples.

                If needed, sample with replacement.

                Form balanced batch of up to 128 items.


                Standardize:

                    z_s =
                        (z - μ)
                        /
                        σ


                Split:

                    z_aud^s = z_s[0:2048]
                    z_txt^s = z_s[2048:2560]


                Whiten audio:

                    z_aud^w =
                        z_aud^s W_ZCAᵀ


                Encode:

                    a =
                        Dropout(
                            GELU(
                                LN(
                                    W_a z_aud^w + b_a
                                )
                            )
                        )

                    t =
                        Dropout(
                            GELU(
                                LN(
                                    W_t z_txt^s + b_t
                                )
                            )
                        )


                Cross-modal interaction:

                    u =
                        tanh(U_a a)
                        ⊙
                        tanh(U_t t)


                Gated fusion:

                    h =
                        a
                        +
                        sigmoid(gate) ⊙ u


                Joint embedding:

                    e = MLP(h)

                    e ∈ R^256


                ------------------------------------------------
                Branch 1: binary head
                ------------------------------------------------

                    l_head =
                        head(e)

                    L_head =
                        BCEWithLogits(l_head, y)


                ------------------------------------------------
                Branch 2: metric
                ------------------------------------------------

                    e_hat =
                        normalize(e)

                    d_0 =
                        ||e_hat - p_0||²

                    d_a =
                        ||e_hat - p_a||²

                    d_1 =
                        -log Σ_a exp(-d_a)

                    l_metric =
                        4.0(d_0 - d_1)

                    L_proto =
                        BCEWithLogits(
                            l_metric,
                            y
                        )


                ------------------------------------------------
                Branch 3: CLAP alignment
                ------------------------------------------------

                    t_bona, t_forg =
                        learned text anchors
                        conditioned by local dsp_bar

                    l_align =
                        exp(
                            clamp(
                                logit_scale,
                                max=4.6
                            )
                        )
                        ×
                        cosine(e, t_c)

                    L_align =
                        CrossEntropy(
                            l_align,
                            y
                        )


                ------------------------------------------------
                Global mean supervision
                ------------------------------------------------

                    For every global class mean μ_c:

                        e_c =
                            local_model(μ_c)

                        L_mean_head =
                            BCEWithLogits(
                                head(e_c),
                                y_c
                            )

                        L_mean_metric =
                            BCEWithLogits(
                                metric(e_c),
                                y_c
                            )

                    L_mean =
                        L_mean_head
                        +
                        L_mean_metric


                ------------------------------------------------
                Game-theoretic loss
                ------------------------------------------------

                    For every attack mean a:

                        p_H(a) =
                            head_probability(a)

                        p_A(a) =
                            alignment_probability(a)

                        m_H(a) =
                            p_H(a)
                            -
                            p_H(bonafide)

                        m_A(a) =
                            p_A(a)
                            -
                            p_A(bonafide)

                        q_a =
                            sigmoid(
                                10(
                                    m_H(a)-0.30
                                )
                            )

                        p_a =
                            sigmoid(
                                10(
                                    m_A(a)-0.30
                                )
                            )

                        w_a =
                            stop_gradient(
                                1-q_a
                            )

                        U_a =
                              p_a q_a (3.0)
                            + p_a(1-q_a)(0.0)
                            + (1-p_a)q_a(5.0)
                            + (1-p_a)(1-q_a)(1.0)

                    L_game =
                        mean_a[
                            w_a(-U_a)
                        ]


                ------------------------------------------------
                FedProx
                ------------------------------------------------

                    L_FedProx =
                        0.5 × 1e-3 ×
                        ||θ_local - θ_global||²


                ------------------------------------------------
                Total loss
                ------------------------------------------------

                    L_total =
                          L_head
                        + 0.3 L_align
                        + 0.4 L_proto
                        + 0.6 L_mean
                        + w_game L_game
                        + L_FedProx


                ------------------------------------------------
                Parameter update
                ------------------------------------------------

                    optimizer.zero_grad()

                    L_total.backward()

                    clip gradients:
                        max_norm = 5.0

                    AdamW.step()


        END local epochs


        Upload:

            local_model parameters

    END clients


    SERVER
        Optionally apply DP clipping/noise.

        Compute normalized FedAvg weights:

            w_k =
                |D_k|
                /
                Σ_j |D_j|

        Update:

            θ_(r+1) =
                Σ_k w_k θ_k


END rounds


FINAL MODEL
    θ* ← θ_R


FINAL PROTOTYPES
    For every bank mean:

        e_c =
            g_{θ*}(μ_c)

        p_c =
            normalize(e_c)

    Return:

        p_0
        {p_A01, ..., p_A19}

```

---

# 59. Local Client Procedure in Plain Language

A single client therefore performs:

```text
1. Take only its own cached examples.
2. Keep only N real forged examples for each locally owned attack.
3. Receive the current global model.
4. Receive the fixed global normalization and ZCA transform.
5. Use global class means for attacks it does not own.
6. Replace global means with local means for classes it does own.
7. Build current prototypes.
8. Form balanced 50/50 bona-fide/forgery mini-batches.
9. Compute the three decision losses.
10. Use global mean supervision.
11. Use game-theoretic weighting.
12. Use FedProx to remain close to the global model.
13. Backpropagate.
14. Clip gradients at 5.0.
15. Update with AdamW.
16. Return the updated parameters to the server.
```

The client never requires another client's waveform.

---

# 60. Server Procedure in Plain Language

The server performs:

```text
1. Keep the current global model.
2. Send it to participating clients.
3. Receive the updated local parameters.
4. Optionally apply upload clipping/noise.
5. Weight clients by local sample count.
6. Apply FedAvg parameter aggregation.
7. Produce the next global model.
8. Start the next round.
```

In the current reference script, the statistics bank is constructed before the round loop rather than being rebuilt after every communication round.

Thus the exact computational behavior is:

```text
initial statistics bank
        ↓
fixed class means
        ↓
current global model changes each round
        ↓
prototype embeddings change because g_θ changes
```

---

# 61. Inference on a New Audio Recording

After the final global model:

```text
θ*
```

has been obtained, inference is completely feed-forward.

For a new recording:

```text
audio
  ↓
mono / 16 kHz / 4 s
  ↓
waveform normalization
  ↓
XLS-R
  ↓
2048-D audio feature
+
24-D DSP descriptor
  ↓
deterministic DSP sentence
  ↓
CLAP text
  ↓
512-D text feature
  ↓
2560-D multimodal feature
  ↓
standardization
  ↓
audio ZCA
  ↓
gated multimodal fusion
  ↓
256-D embedding
```

No federated communication is required during inference.

---

# 62. Inference Branch 1 — Binary Head

```text
l_head =
    w_hᵀe + b_h
```

then:

```text
p_head =
    sigmoid(l_head)
```

---

# 63. Inference Branch 2 — Prototype Metric

Normalize:

```text
e_hat =
    e / ||e||₂
```

Compute:

```text
d_0 =
    ||e_hat - p_0||₂²
```

and:

```text
d_a =
    ||e_hat - p_a||₂²
```

then:

```text
d_1 =
    -log(
        Σ_a exp(-d_a)
    )
```

and:

```text
p_metric =
    sigmoid(
        4.0(d_0 - d_1)
    )
```

---

# 64. Inference Branch 3 — CLAP Alignment

The code uses:

```text
text_anchors(None)
```

at inference.

Therefore the inference anchors are:

```text
learned class prompts
```

without passing a new per-recording DSP vector into the prompt-conditioning network.

The CLAP logits are:

```text
s_c =
    exp(
        clamp(
            logit_scale,
            max=4.6
        )
    )
    e_hatᵀ t_c
```

and:

```text
p_align =
    softmax([s_bona, s_forg])[FORGERY]
```

---

# 65. Final Score Fusion

The three probabilities are combined using fixed weights:

```text
FUSE_HEAD   = 0.50
FUSE_METRIC = 0.35
FUSE_ALIGN  = 0.15
```

so:

```text
p_forgery =
      0.50 p_head
    + 0.35 p_metric
    + 0.15 p_align
```

The weights are fixed and are not learned by backpropagation.

They satisfy:

```text
0.50 + 0.35 + 0.15 = 1.0
```

so the final result remains interpretable as a weighted probability-like score.

---

# 66. Final Decision

The final detector returns:

```text
p_forgery ∈ [0,1]
```

and converts it to a binary decision using the operating threshold:

```text
if p_forgery >= τ:
    FORGERY
else:
    BONAFIDE
```

The evaluation code subsequently converts the score to an ASV-style score with:

```text
higher score = more bona fide
```

so the metric evaluation uses the negative forgery score.

The threshold is therefore an operating-point parameter rather than another trainable network parameter.

---

# 67. One-Line View of the Complete Detector

```text
Audio
→ mono + 16 kHz + 4 s
→ waveform normalization
→ frozen XLS-R
→ mean + std pooling
→ 2048-D speech feature
→ 24-D DSP descriptor
→ deterministic DSP-to-text
→ frozen CLAP text embedding
→ 512-D semantic signal feature
→ concatenate to 2560-D
→ global standardisation
→ 2048-D audio ZCA whitening
→ 512-D audio encoder
→ 512-D text encoder
→ gated low-rank bilinear fusion
→ 256-D embedding
→ binary head
→ per-attack prototype metric
→ CLAP semantic alignment
→ mean supervision + game-aware federated training
→ FedProx
→ AdamW local updates
→ FedAvg server aggregation
→ weighted score fusion
→ final forgery probability
```

---

# 68. Client–Server Interaction at a Glance

```text
                           ┌─────────────────────────┐
                           │          SERVER         │
                           │                         │
                           │ Global model θ_r        │
                           │ Global μ, σ             │
                           │ ZCA matrix              │
                           │ Statistics bank         │
                           │ Attack means             │
                           └────────────┬────────────┘
                                        │
                               broadcast global model
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             │                          │                          │
             ▼                          ▼                          ▼
      ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
      │   Client 1  │           │   Client 2  │           │   Client K  │
      │             │           │             │           │             │
      │ local bona  │           │ local bona  │           │ local bona  │
      │ + own few-  │           │ + own few-  │           │ + own few-  │
      │ shot attacks│           │ shot attacks│           │ shot attacks│
      │             │           │             │           │             │
      │ N samples / │           │ N samples / │           │ N samples / │
      │ owned attack│           │ owned attack│           │ owned attack│
      └──────┬──────┘           └──────┬──────┘           └──────┬──────┘
             │                          │                          │
             │ local AdamW              │ local AdamW              │ local AdamW
             │ + FedProx               │ + FedProx               │ + FedProx
             │ + mean supervision      │ + mean supervision      │ + mean supervision
             │ + game loss             │ + game loss             │ + game loss
             ▼                          ▼                          ▼
       local θ_1                   local θ_2                   local θ_K
             │                          │                          │
             └───────────────┬──────────┴──────────┬──────────────┘
                             │                     │
                             │      model uploads  │
                             ▼                     ▼
                       ┌────────────────────────────────┐
                       │             SERVER             │
                       │                                │
                       │ sample-weighted FedAvg          │
                       │ optional DP clipping/noise     │
                       │                                │
                       │ θ_(r+1) = Σ_k w_k θ_k          │
                       └────────────────────────────────┘
```

---

# 69. Privacy Boundary

```text
RAW AUDIO
    → stays local

INDIVIDUAL WAVEFORM
    → stays local

INDIVIDUAL DSP DESCRIPTOR
    → stays local

INDIVIDUAL CACHED MULTIMODAL FEATURE
    → stays local

LOCAL MODEL UPDATE
    → uploaded

OPTIONAL NOISY/CLIPPED UPDATE
    → uploaded when DP is enabled

GLOBAL MODEL
    → broadcast back to clients
```

The class-statistics mechanism provides a separate information-sharing path based on aggregate feature statistics rather than raw utterances.

---

# 70. Parameter Lifecycle Summary

```text
PARAMETER / STATE             INITIALIZED FROM          UPDATED HOW?
--------------------------------------------------------------------------------
XLS-R weights                 pretrained checkpoint    never updated
CLAP weights                  pretrained checkpoint    never updated
Tokenizer                     pretrained tokenizer     never updated

μ                             cached feature mean      fixed
σ                             cached feature std       fixed
W_ZCA                         covariance eigendecomp    fixed

Audio Linear                  PyTorch default init     AdamW
Text Linear                   PyTorch default init     AdamW
LayerNorm                     PyTorch default init     AdamW
U_a                           PyTorch default init     AdamW
U_t                           PyTorch default init     AdamW
Gate                          constant -1.0            AdamW

Output projection             PyTorch default init     AdamW
Binary head                   PyTorch default init     AdamW
Text projection               PyTorch default init     AdamW
logit_scale                   constant 2.3             AdamW

Soft context                  N(0, 0.02²)             AdamW
Meta trunk                    PyTorch default init     AdamW
Meta-soft head                PyTorch default init     AdamW

Adapter context               zeros (fallback)        AdamW
Adapter network               PyTorch default init     AdamW

Class mean μ_c                feature statistics       recomputed from bank
Class variance                feature statistics       stored in bank
Prototype p_c                 g_θ(μ_c)                 recomputed with current θ

FedProx reference             θ_global                 refreshed each round
FedAvg weights                |D_k|/Σ|D_j|            recomputed each round

Game weight                   0.10 → 0.20              warm-up by round
γ metric scale                4.0                      fixed
Game κ                        10.0                     fixed
Game τ_g                      0.30                     fixed

Score fusion weights          0.50/0.35/0.15           fixed
```

---

# 71. Exact Training Hyperparameters

```text
Audio:
    sample rate = 16 kHz
    duration = 4 s
    XLS-R hidden size = 1024
    pooled audio size = 2048

DSP:
    descriptor size = 24
    nfft = 1024
    hop = 256
    f0 range = 70–400 Hz
    spectral roll-off = 0.85
    HF threshold = 4000 Hz
    VHF threshold = 7000 Hz
    modulation bands = 2–8 Hz and 8–20 Hz
    MFCC count = 20

Multimodal:
    audio stream = 2048
    text stream = 512
    total = 2560

Whitening:
    ZCA = enabled
    shrinkage ε_w = 0.10

Fusion:
    encoder width = 512
    embedding width = 256
    dropout = 0.30
    gate initialization = -1.0

Prompt:
    mode = soft
    context tokens = 8
    context initialization std = 0.02
    DSP meta-network = 24 → 128 → text hidden

Federated:
    clients = 4
    rounds = 5
    local epochs = 6
    local batch size = 128
    half-batch per class = 64

Optimization:
    optimizer = AdamW
    learning rate = 1e-3
    weight decay = 1e-4
    gradient clipping = 5.0

FedProx:
    μ = 1e-3

Loss:
    alignment weight = 0.3
    prototype weight = 0.4
    mean-supervision weight = 0.6
    game max weight = 0.20

Game:
    T = 5
    R = 3
    P = 1
    S = 0
    κ = 10
    τ_g = 0.30
    warm-up = 2 rounds

Inference:
    head = 0.50
    metric = 0.35
    alignment = 0.15

Optional DP:
    sigma = 0.0 by default
    clip = 1.0
```

---

# 72. What Changes During One Complete Federated Round?

A useful way to track parameter evolution is:

```text
START ROUND r
    │
    ├── global θ_r exists
    │
    ├── copy θ_r → each client
    │
    ├── create fresh AdamW optimizer per client
    │
    ├── local forward/backward steps
    │      │
    │      ├── θ changes
    │      ├── gate changes
    │      ├── head changes
    │      ├── prompt changes
    │      ├── text projection changes
    │      └── logit scale changes
    │
    ├── local prototypes are recomputed using new θ
    │
    ├── gradients are clipped
    │
    ├── local model uploaded
    │
    └── server averages client models
                ↓
            global θ_(r+1)
```

Meanwhile:

```text
XLS-R        = unchanged
CLAP         = unchanged
μ            = unchanged
σ            = unchanged
W_ZCA        = unchanged
DSP formulas = unchanged
prototype vectors themselves = recomputed, not independently optimized
```

---

# 73. Current Reference Implementation Note

The methodology is designed as a federated few-shot detector, but the supplied reference script is a **simulated federation** rather than a deployed cross-silo communication system.

In the current script:

```text
1. Features are extracted/cached centrally.
2. Global normalization statistics are computed from the cache.
3. ZCA is computed once from standardized cached audio features.
4. The training pool is then formed.
5. Attack families are partitioned among simulated clients.
6. A statistics bank is built once before the round loop.
7. Each round performs local model training.
8. The server performs FedAvg on model parameters.
```

This means the current reproduction script should not be described as a production-ready privacy-preserving deployment without qualification.

For a strict real deployment, the corresponding statistics:

```text
μ
σ
Σ z
Σ z²
class means
```

should be generated on the clients and aggregated by the server before being broadcast.

Likewise, evaluation/test features should not be used when constructing global preprocessing statistics.

This distinction is important for faithfully reproducing the current experiment versus implementing the protocol as a real cross-silo federated system.

---

# 74. Code-Faithful Summary

The complete FeMMA learning mechanism can be summarized as:

```text
1. Normalize the raw waveform.
2. Extract frozen XLS-R mean/std features.
3. Extract 24 explicit DSP descriptors.
4. Convert DSP values to deterministic text.
5. Encode the text with frozen CLAP.
6. Concatenate speech and CLAP-DSP features.
7. Standardize the 2560-D vector.
8. ZCA-whiten only the 2048-D audio block.
9. Project audio and text into a common 512-D space.
10. Form multiplicative cross-modal interactions.
11. Inject them through a learnable negative-initialized gate.
12. Produce a 256-D joint embedding.
13. Train a primary binary classifier.
14. Train a per-attack prototype metric.
15. Train CLAP semantic alignment.
16. Build global class means from aggregate statistics.
17. Use those means to supervise globally unseen attack families.
18. Use a game-theoretic loss to emphasize attacks where the head is weak.
19. Use FedProx to constrain local drift.
20. Update each client with AdamW.
21. Clip local gradients at 5.0.
22. Aggregate client parameters by sample-weighted FedAvg.
23. Repeat for R communication rounds.
24. Build final attack-specific prototypes from the final encoder.
25. At inference, obtain three independent forgery probabilities.
26. Fuse:
       0.50 × head
     + 0.35 × metric
     + 0.15 × alignment
27. Convert the fused score to BONAFIDE or FORGERY using the selected threshold.
```

---

# 75. Final Detection Equation

For a new audio recording `x`:

```text
e = FeMMA_Encoder(
        Standardize(
            XLSR(x)
            ||
            CLAP(
                DSP_TO_TEXT(
                    DSP(x)
                )
            )
        )
    )
```

Then:

```text
p_head   = sigmoid(
              head(e)
           )
```

```text
p_metric =
    sigmoid(
        4.0
        (
            ||ê - p_0||²
            -
            [-log Σ_a exp(-||ê-p_a||²)]
        )
    )
```

where:

```text
ê = e / ||e||₂
```

and:

```text
p_align =
    softmax(
        τ [
            êᵀt_bona,
            êᵀt_forg
        ]
    )[FORGERY]
```

with:

```text
τ =
    exp(
        clamp(
            logit_scale,
            max=4.6
        )
    )
```

Finally:

```text
p_forgery =
      0.50 p_head
    + 0.35 p_metric
    + 0.15 p_align
```

and:

```text
prediction =
    FORGERY      if p_forgery ≥ τ_decision
    BONAFIDE     otherwise
```

The result is a detector that combines:

```text
learned speech evidence
+
explicit acoustic evidence
+
semantic signal description
+
cross-modal interaction
+
attack-specific geometry
+
global few-shot statistics
+
adaptive branch alignment
+
federated optimization
```

into a single final forgery decision.
