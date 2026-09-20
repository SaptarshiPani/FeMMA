# FeMMA: Federated Few-Shot Audio Deepfake Detection

FeMMA is a binary audio deepfake detector designed for the setting where different organizations hold different attack families and only a few forged examples are available for each family. Training is performed in a federated manner: raw audio remains with the client that owns it, while model parameters and aggregate feature statistics are exchanged with a central server.

The detector combines three complementary views of an utterance:

* **Discriminative view:** a binary classifier operating on the learned joint embedding.
* **Attack-aware metric view:** distances to global bona-fide and per-attack prototypes.
* **Semantic view:** alignment between the audio embedding and CLAP-based class anchors.

The final decision is obtained by fusing the three forgery scores.

---

## 1. End-to-End Pipeline

```text
                         INPUT AUDIO
                              │
                              ▼
                  16-kHz / mono / 4-sec crop
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
      Frozen XLS-R                    24-D Signal Descriptor
      (Wav2Vec 2.0)                           │
             │                                 ▼
             │                         Deterministic text
             │                         description of signal
             │                                 │
             │                                 ▼
             │                         Frozen CLAP Text
             │                                 │
             └──────────────┬──────────────────┘
                            ▼
                [ Audio || CLAP-Text ]
                       2560-D
                            │
                            ▼
                Standardisation + ZCA
                            │
                  ┌─────────┴─────────┐
                  │                   │
                  ▼                   ▼
             Audio branch         Text branch
                  │                   │
                  └─────────┬─────────┘
                            ▼
                 Gated cross-modal fusion
                            │
                            ▼
                       256-D embedding
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
    Binary Head       Prototype Metric    CLAP Alignment
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                   Score-level fusion
                            │
                            ▼
                 FORGERY PROBABILITY
                            │
                            ▼
                  BONAFIDE / FORGERY
```

The frozen XLS-R and CLAP encoders are used only for representation
extraction. The federated optimisation acts on the downstream fusion and
decision modules.

---

# 2. Federated Few-Shot Setting

Let each training example be

```text
(x_i, y_i, a_i)
```

where:

```text
x_i : waveform
y_i : 0 = bona fide, 1 = forgery
a_i : attack family, when y_i = 1
```

For client `k`:

```text
D_k = D_k^bonafide ∪ ⋃_{a∈A_k} D_{k,a}^forgery
```

`A_k` is the subset of attack families available at that client.

The important part is that the clients are **non-IID**:

```text
Client 1 → bona fide + a subset of attacks
Client 2 → bona fide + a different subset of attacks
Client 3 → bona fide + another subset of attacks
...
```

For each locally owned attack family, only `N` forged samples are retained.
Thus, an attack may be represented by only a handful of real examples at its
owner.

No forged samples are copied to other clients and no synthetic samples are
created.

The paper's main federated experiments use `K = 4` and `N = 5`; the reference
implementation exposes the client count and shot budget as configurable
parameters.
-----------

# 3. Feature Extraction

Feature extraction is performed once and cached because the two backbone
networks are frozen.

## 3.1 Audio representation

The waveform is converted to mono, resampled to 16 kHz, and cropped or padded
to 4 seconds.

The waveform is passed through the frozen XLS-R encoder:

```text
H = [h_1, h_2, ..., h_T]
```

with 1024-dimensional hidden states.

Instead of keeping the entire sequence, FeMMA stores its temporal mean and
standard deviation:

```text
z_aud = [ mean(H) || std(H) ]
```

giving:

```text
z_aud ∈ R^2048
```

The implementation uses this representation as the cached speech feature.

---

## 3.2 Signal descriptor

A 24-dimensional signal descriptor is extracted from the waveform:

```text
Pitch:
    F0 mean
    F0 standard deviation
    jitter
    voiced fraction
    F0 smoothness

Voice quality:
    shimmer
    HNR
    CPP

Spectral shape:
    spectral centroid
    bandwidth
    flatness
    roll-off

Spectral distribution:
    entropy
    skewness
    kurtosis
    spectral contrast

Temporal / phase cues:
    zero-crossing statistics
    high-frequency energy
    very-high-frequency energy
    group-delay variance
    modulation energy
    MFCC-delta variance
```

The complete descriptor is

```text
d_i ∈ R^24
```

and is computed deterministically from the waveform.

---

## 3.3 Signal-to-text representation

The 24-D descriptor is converted into a deterministic textual description,
for example:

```text
"speech with moderate pitch variation, low jitter,
weak harmonic structure, high spectral flatness, ..."
```

The description is then encoded using the frozen CLAP text tower:

```text
z_txt = CLAP_text(description(d))
```

with

```text
z_txt ∈ R^512
```

This turns explicitly measured acoustic properties into a semantic
representation without using free-form text generation.

---

## 3.4 Cached multimodal feature

The two representations are concatenated:

```text
z = [ z_aud || z_txt ]
```

so that:

```text
z ∈ R^2560
```

This cached feature is the input to the trainable federated model.

---

# 4. Standardisation and ZCA Whitening

Because the audio and text features have different scales and are correlated,
the concatenated representation is first standardised:

```text
z_s = (z - μ) / (σ + ε)
```

where `μ` and `σ` are estimated from training data.

The representation is then split into:

```text
z_aud^s = first 2048 dimensions
z_txt^s = last 512 dimensions
```

Only the audio stream is whitened.

Let

```text
C = covariance(z_aud^s)
C = V Λ Vᵀ
```

Then the regularised ZCA matrix is

```text
W_ZCA = V (Λ + ε_w I)^(-1/2) Vᵀ
```

and

```text
z_aud^w = W_ZCA z_aud^s
```

The ZCA step decorrelates the audio representation while retaining its
original coordinate orientation.

---

# 5. Gated Cross-Modal Fusion

The two streams are projected independently:

```text
h_aud = Dropout(GELU(LN(W_a z_aud^w + b_a)))

h_txt = Dropout(GELU(LN(W_t z_txt^s + b_t)))
```

The model then forms a low-rank multiplicative interaction:

```text
u =
    tanh(U_a h_aud)
    ⊙
    tanh(U_t h_txt)
```

where `⊙` denotes element-wise multiplication.

A learnable gate determines how much of this interaction is injected into the
audio path:

```text
h = h_aud + sigmoid(β) ⊙ u
```

The gate is initialised negatively so that the audio stream dominates at the
start of training. The fused representation is finally mapped to:

```text
e = MLP(h)

e ∈ R^256
```

This 256-D vector is the representation used by all three detection branches.

---

# 6. Federated Statistics Bank

The main difficulty with attack-disjoint clients is that a client cannot
directly train on attack families that it does not own.

FeMMA addresses this through aggregate class statistics.

For each class `c`, a client computes:

```text
S_c  = Σ_i z_i^s
SS_c = Σ_i (z_i^s ⊙ z_i^s)
N_c  = number of samples
```

The server aggregates these quantities:

```text
S_c^global  = Σ_k S_{k,c}
SS_c^global = Σ_k SS_{k,c}
N_c^global  = Σ_k N_{k,c}
```

and obtains the global class mean:

```text
μ_c = S_c^global / N_c^global
```

The resulting bank contains:

```text
μ_bonafide
μ_A01
μ_A02
...
μ_A19
```

for the attack families present in the federation.

Only aggregate statistics are used; the raw utterances are not transferred.
These global means are then broadcast back to the clients and embedded by the
current global model.
---------------------

# 7. Attack-Aware Prototypes

For every class mean:

```text
e_c = g_θ(μ_c)
```

and its normalised prototype is

```text
p_c = e_c / ||e_c||_2
```

This gives one prototype for bona fide speech and one prototype for each
attack family.

The important point is that the forged class is **not represented by a single
global spoof centroid**.

Instead:

```text
BONAFIDE → p_0

FORGERY → {p_A01, p_A02, ..., p_A19}
```

The per-attack structure is what lets the metric branch distinguish different
forgery modes.

---

# 8. Three Detection Branches

## 8.1 Discriminative branch

The joint embedding is passed through a binary head:

```text
l_head = w_hᵀ e + b_h
p_head = sigmoid(l_head)
```

The loss is binary cross-entropy:

```text
L_head =
    -y log(p_head)
    -(1-y) log(1-p_head)
```

This is the primary classification branch.

---

## 8.2 Attack-aware prototype branch

First normalise the embedding:

```text
e_hat = e / ||e||_2
```

Distance to the bona-fide prototype:

```text
d_0 = ||e_hat - p_0||_2²
```

Distance to each attack prototype:

```text
d_a = ||e_hat - p_a||_2²
```

The attack-side distance is formed with a differentiable soft minimum:

```text
d_1 = -log Σ_a exp(-d_a)
```

The metric logit is:

```text
l_metric = γ(d_0 - d_1)
```

and the corresponding forgery probability is:

```text
p_metric = sigmoid(l_metric)
```

Intuitively:

```text
far from bona fide
+
close to at least one attack prototype
        ↓
higher forgery score
```

---

## 8.3 CLAP alignment branch

Two class-specific semantic anchors are learned for:

```text
BONAFIDE
FORGERY
```

The anchors are projected into the same embedding space and compared with the
normalised utterance embedding:

```text
s_c = exp(η) e_hatᵀ t_c
```

A two-class softmax gives the semantic prediction:

```text
p_align = P(FORGERY | e)
```

The alignment branch is trained with cross-entropy against the binary label.
The signal descriptor also conditions the CLAP prompt, allowing the semantic
representation to depend on the acoustic characteristics of the utterance.

---

# 9. Global Mean Supervision

The global class means are also passed through the current model:

```text
e_c = g_θ(μ_c)
```

and supervised with the same binary head and prototype metric used for real
training examples.

Thus, a client that never sees attack `A17` locally can still receive
supervision from the aggregated `μ_A17`.

This is the main mechanism used to reduce the loss of attack coverage caused
by the non-IID federation.

---

# 10. Game-Theoretic Alignment

The discriminative head and CLAP alignment branch may behave differently on
different attack families.

For an attack `a`, let:

```text
p_H(a) = head forgery probability
p_A(a) = alignment forgery probability
```

Their separation from the bona-fide class is measured by:

```text
m_H(a) = p_H(a) - p_H(0)
m_A(a) = p_A(a) - p_A(0)
```

These margins are converted to soft cooperation scores:

```text
q_a = sigmoid(κ(m_H(a) - τ_g))

p_a = sigmoid(κ(m_A(a) - τ_g))
```

The two branches are then treated as players in a soft Prisoner's Dilemma.

Using the payoff ordering

```text
T > R > P > S
```

the expected utility for attack `a` is

```text
U_a =
    p_a q_a R
  + p_a (1-q_a) S
  + (1-p_a) q_a T
  + (1-p_a)(1-q_a) P
```

and the game loss is

```text
L_game =
    1/|A| Σ_a stop_gradient(1-q_a) (-U_a)
```

The factor `(1-q_a)` is the key part.

If the discriminative branch already handles an attack well, the game term
becomes small.

If the head is weak on that attack, the alignment branch receives stronger
pressure.

In other words, the game mechanism directs additional learning toward attack
families that remain difficult rather than reinforcing already easy cases.

The game loss is warmed up for the first two communication rounds to avoid
using unreliable early-stage margins.

---

# 11. Local Client Objective

For a local mini-batch, the training objective is:

```text
L_total =
      L_head
    + λ_align L_align
    + λ_proto L_proto
    + λ_mean  L_mean
    + λ_game  L_game
    + L_FedProx
```

where:

```text
λ_align = 0.3
λ_proto = 0.4
λ_mean  = 0.6
```

and `λ_game` is gradually increased to its configured maximum.

The FedProx term keeps the local model close to the model received from the
server:

```text
L_FedProx =
    μ_FedProx / 2 · ||θ_local - θ_global||²
```

with the reference configuration using:

```text
μ_FedProx = 1e-3
```

---

# 12. What Happens on Each Client

A client performs the following operations entirely locally:

```text
1. Load its own audio.
2. Generate/cache XLS-R and CLAP features.
3. Keep only its assigned few-shot attack samples.
4. Receive the current global model.
5. Receive global μ, σ, ZCA and class means.
6. Build the current attack-aware prototypes.
7. Train on its local real examples.
8. Add mean-based supervision from the global statistics bank.
9. Apply the game-theoretic regularisation.
10. Apply FedProx.
11. Update the trainable parameters with AdamW.
12. Send the updated model/statistics to the server.
```

At no point does the client require another client's waveform or individual
training sample.

---

# 13. What Happens on the Server

At the end of each local training phase, the server:

```text
1. Receives the client updates.
2. Optionally clips/noises the uploaded updates.
3. Aggregates the client models using sample-weighted FedAvg.
4. Aggregates class statistics into the global statistics bank.
5. Rebuilds the global attack means.
6. Updates the global model.
7. Broadcasts the new model and global statistics.
8. Starts the next communication round.
```

The parameter aggregation is:

```text
θ^(r+1) =
    Σ_k [ |D_k| / Σ_j |D_j| ] θ_k^(r+1)
```

so clients contribute according to their local training-set size.

---

# 14. Complete Training Algorithm

```text
INPUT
    K              number of clients
    N              forged shots per attack family
    R              federated rounds
    E              local epochs
    D_1,...,D_K    non-IID client datasets
    XLS-R          frozen speech encoder
    CLAP           frozen text encoder

OUTPUT
    θ*             final global detector
    P*             global attack-aware prototypes


PREPROCESSING
    For every training utterance x:

        x ← mono(x)
        x ← resample(x, 16 kHz)
        x ← crop_or_pad(x, 4 s)

        H ← XLS-R(x)

        z_aud ← [mean(H) || std(H)]

        d ← DSP_DESCRIPTOR(x)

        q ← DSP_TO_TEXT(d)

        z_txt ← CLAP_TEXT(q)

        z ← [z_aud || z_txt]

    Cache z and d.


FEDERATED INITIALISATION
    Initialise global trainable parameters θ_0.

    Estimate global training statistics:

        μ
        σ
        W_ZCA

    Aggregate class sufficient statistics and compute:

        μ_c

    Broadcast:

        θ_0, μ, σ, W_ZCA, {μ_c}


FOR round r = 1,...,R:

    SERVER
        Broadcast current global state.

    FOR each client k:

        θ_k ← θ_r

        Build local few-shot training set.

        FOR epoch e = 1,...,E:

            Sample balanced local mini-batch.

            Standardise features:

                z_s = (z - μ) / (σ + ε)

            Split:

                z_aud^s
                z_txt^s

            Whiten audio:

                z_aud^w = W_ZCA z_aud^s

            Encode:

                h_aud = Enc_audio(z_aud^w)
                h_txt = Enc_text(z_txt^s)

            Fuse:

                u = tanh(U_a h_aud)
                    ⊙
                    tanh(U_t h_txt)

                h = h_aud + sigmoid(β) ⊙ u

                e = MLP(h)

            Compute:

                L_head
                L_align
                L_proto

            Embed global class means and compute:

                L_mean

            Evaluate head/alignment margins on
            global attack means and compute:

                L_game

            Add FedProx:

                L_FedProx

            Total loss:

                L_total =
                    L_head
                  + λ_align L_align
                  + λ_proto L_proto
                  + λ_mean  L_mean
                  + λ_game  L_game
                  + L_FedProx

            Backpropagate.

            Clip gradients.

            Update θ_k with AdamW.

        END FOR

        Upload:

            θ_k
            aggregated class statistics

    END FOR

    SERVER

        Aggregate client statistics.

        Aggregate model parameters:

            θ_(r+1) =
                Σ_k w_k θ_k

        Update global class means/prototypes.

END FOR


Return:

    θ*
    {p_0*, p_A01*, ..., p_A19*}
```

---

# 15. Inference on a New Audio Recording

Training is complete after the final global model is obtained.

For a new recording:

```text
audio
  ↓
preprocessing
  ↓
XLS-R feature + DSP descriptor
  ↓
CLAP signal-text feature
  ↓
2560-D multimodal feature
  ↓
standardisation + audio ZCA
  ↓
gated fusion encoder
  ↓
256-D embedding
```

The three branches are then evaluated independently.

### Head score

```text
p_head = sigmoid(w_hᵀe + b_h)
```

### Prototype score

```text
d_0 = ||e_hat - p_0||²

d_a = ||e_hat - p_a||²

d_1 = -log Σ_a exp(-d_a)

p_metric = sigmoid(γ(d_0 - d_1))
```

### CLAP alignment score

```text
p_align = P(FORGERY | e)
```

Finally:

```text
p_forgery =
      0.50 p_head
    + 0.35 p_metric
    + 0.15 p_align
```

The reference implementation uses these three weights directly.
The final prediction is:

```text
if p_forgery >= τ:
    FORGERY
else:
    BONAFIDE
```

where `τ` is the operating threshold selected on a development/calibration
set.

---

# 16. One-Line View of the Complete Detector

```text
Audio
→ XLS-R + DSP
→ CLAP signal description
→ multimodal feature
→ standardisation + ZCA
→ gated fusion
→ joint embedding
→ {binary head + attack prototypes + CLAP anchors}
→ game-aware training
→ weighted score fusion
→ final forgery probability
```

---

# 17. Client–Server Interaction at a Glance

```text
                        ┌───────────────────────┐
                        │         SERVER        │
                        │                       │
                        │ Global model θ        │
                        │ Global statistics     │
                        │ Attack means          │
                        │ Attack prototypes     │
                        └───────────┬───────────┘
                                    │
                            broadcast θ + stats
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
             ▼                      ▼                      ▼
       ┌───────────┐          ┌───────────┐          ┌───────────┐
       │ Client 1  │          │ Client 2  │          │ Client K  │
       │           │          │           │          │           │
       │ local     │          │ local     │          │ local     │
       │ bona fide │          │ bona fide │          │ bona fide │
       │ + attacks │          │ + attacks │          │ + attacks │
       │           │          │           │          │           │
       │ N shots / │          │ N shots / │          │ N shots / │
       │ attack    │          │ attack    │          │ attack    │
       └─────┬─────┘          └─────┬─────┘          └─────┬─────┘
             │                      │                      │
             │ local optimisation   │                      │
             └──────────────┬───────┴──────────────┬───────┘
                            │                      │
                     model updates        aggregate statistics
                            │                      │
                            ▼                      ▼
                        ┌───────────────────────────┐
                        │          SERVER           │
                        │                           │
                        │  FedAvg                  │
                        │  statistics aggregation  │
                        │  prototype refresh       │
                        └───────────────────────────┘
```

The intended privacy boundary is simple:

```text
RAW AUDIO                     → stays at client
INDIVIDUAL UTTERANCES         → stay at client
MODEL UPDATES                 → sent to server
AGGREGATE FEATURE STATISTICS  → sent to server
GLOBAL MODEL                  → returned to clients
```

---

# 18. Implementation Note

The conceptual method is federated at the feature-statistics and optimisation
levels. The current reference script is a simulated federation and computes
the cached global normalisation/ZCA statistics before constructing the client
training partitions. For a strict deployment implementation, these statistics
should instead be formed from client-side sufficient statistics and aggregated
by the server, with no evaluation/test features involved.

This distinction is important when reproducing the experiments versus
deploying the protocol in a real cross-silo federated environment.
