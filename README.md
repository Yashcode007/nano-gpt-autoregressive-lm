# 🧠 nanoGPT: Autoregressive Character-Level Language Model

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Architecture-Decoder--Only%20Transformer-blueviolet?style=for-the-badge)]()
[![Dataset](https://img.shields.io/badge/Dataset-Tiny%20Shakespeare-success?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)]()

A clean, self-contained, from-scratch PyTorch implementation of a **nanoGPT-style Decoder-Only Autoregressive Transformer** trained on the **Tiny Shakespeare** dataset. 

This repository covers the complete deep generative modeling pipeline: custom character tokenization, causal self-attention with mathematical causality verification, Pre-LayerNorm residual transformer blocks, weight tying, training with AdamW, and sampling strategies (temperature scaling and top-$k$ filtering).

Developed as part of the **Advances in Deep Generative Modeling** course (UT Austin).

---

## 📌 Table of Contents

- [Architectural Overview](#-architectural-overview)
- [Model Specifications](#-model-specifications)
- [Mathematical Foundations & Key Components](#-mathematical-foundations--key-components)
  - [1. Character Tokenizer & Dataset Split](#1-character-tokenizer--dataset-split)
  - [2. Explicit Causal Self-Attention](#2-explicit-causal-self-attention)
  - [3. Mathematical Verification of Causality](#3-mathematical-verification-of-causality)
  - [4. Feed-Forward Network (MLP) & Pre-LN Transformer Blocks](#4-feed-forward-network-mlp--pre-ln-transformer-blocks)
  - [5. Weight Tying & Weight Initialization](#5-weight-tying--weight-initialization)
- [Training Dynamics & Results](#-training-dynamics--results)
- [Autoregressive Text Generation & Decoding Experiments](#-autoregressive-text-generation--decoding-experiments)
  - [Temperature Scaling ($T=0.7$ vs $T=1.2$)](#temperature-scaling-t07-vs-t12)
  - [Top-$k$ Sampling (Full Vocab vs Top-10)](#top-k-sampling-full-vocab-vs-top-10)
- [Repository Structure](#-repository-structure)
- [Quickstart & Installation](#-quickstart--installation)
- [Citation & References](#-citation--references)

---

## 🏗️ Architectural Overview

```
Input Tokens (B, T)
        │
        ├──► Token Embedding (wte) ───────────────┐
        │                                         │
        └──► Positional Embedding (wpe) ──────────┼──► Dropout ──► (B, T, C)
                                                  │
 ┌────────────────────────────────────────────────┴────────────────────────┐
 │ Transformer Block (Repeated N_LAYER = 4 times)                          │
 │                                                                         │
 │   Input (x) ────────────────────────┬───────────────────────────────┐   │
 │                                     ▼                               │   │
 │                                LayerNorm (ln_1)                     │   │
 │                                     ▼                               │   │
 │                            CausalSelfAttention                      │   │
 │                                     ▼                               │   │
 │                             Residual Addition (x + Attn) ◄──────────┘   │
 │                                     │                                   │
 │                                     ├───► LayerNorm (ln_2)              │
 │                                     │           ▼                       │
 │                                     │     MLP (4x Expansion + GELU)     │
 │                                     │           ▼                       │
 │                                     └───► Residual Addition ────────┐   │
 └─────────────────────────────────────────────────────────────────────┼───┘
                                                                       ▼
                                                                LayerNorm (ln_f)
                                                                       ▼
                                                              Language Model Head (lm_head)
                                                            [Weight tied to wte: (C -> V)]
                                                                       ▼
                                                                Output Logits (B, T, V)
```

---

## ⚙️ Model Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model Type** | Decoder-Only GPT | Autoregressive Transformer with Pre-LN |
| **Total Parameters** | **807,808** | ~0.81M trainable parameters |
| **Embedding Dimension ($d_{\text{model}}$)** | **128** | Hidden size ($C$) |
| **Number of Layers ($n_{\text{layer}}$)** | **4** | Stacked Transformer blocks |
| **Number of Heads ($n_{\text{head}}$)** | **4** | Multi-head attention heads |
| **Head Dimension ($d_{h}$)** | **32** | $d_{\text{model}} / n_{\text{head}} = 128 / 4$ |
| **Context Length ($\text{block\_size}$)** | **64** | Maximum sequence length ($T$) |
| **MLP Expansion Factor** | **4x** | $4 \times 128 = 512$ hidden units with GELU |
| **Vocabulary Size ($V$)** | **65** | Distinct character tokens in Tiny Shakespeare |
| **Dataset Size** | **1,115,394 chars** | Split 90% Train (1,003,854) / 10% Val (111,540) |
| **Optimizer** | AdamW | Learning rate $= 10^{-3}$, weight decay regularization |
| **Batch Size ($B$)** | **12** | Batch dimension per optimization step |
| **Training Steps** | **1,500 iterations** | Periodic validation loss tracking |

---

## 📐 Mathematical Foundations & Key Components

### 1. Character Tokenizer & Dataset Split
- A character-level vocabulary maps all unique characters in the Shakespeare corpus into integer token indices $0 \le i < V$:
  $$\text{encode}(s): \text{string} \to \mathbb{N}^{|s|}, \quad \text{decode}(l): \mathbb{N}^{|l|} \to \text{string}$$
- **Verification**: Verified lossless bijection $\text{decode}(\text{encode}(s)) = s$.
- **Target Shift Verification**: The next-token prediction targets $y$ are strictly shifted by one position relative to inputs $x$:
  $$x[0, 1:] == y[0, :-1]$$

### 2. Explicit Causal Self-Attention
Implemented with explicit fused $Q, K, V$ projections and scaled dot-product attention:
$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_h}} + M\right) V$$
where $M_{i,j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$ is the lower-triangular causal attention mask.

### 3. Mathematical Verification of Causality
The implementation includes two automated architectural test suites:
1. **Upper-Triangular Mask Verification**: Proves all attention probabilities above the diagonal ($j > i$) in attention weight tensor $P$ are strictly $0.0$:
   $$\max_{j > i} |P_{i,j}| = 0.0$$
2. **Prefix Causality Invariance**: Proves that future token alterations in a sequence suffix do not alter earlier representations:
   $$H_1[:, :m] = H_2[:, :m] \implies \text{Attention}(H_1)[:, :m] \equiv \text{Attention}(H_2)[:, :m]$$
   *(Empirically confirmed: max absolute difference $= 0.00\text{e}+00$)*.

### 4. Feed-Forward Network (MLP) & Pre-LN Transformer Blocks
- **MLP Module**: Linear projection to $4 \times d_{\text{model}}$ (512 dimensions) $\to$ GELU non-linearity $\to$ Linear projection back to $d_{\text{model}}$ (128 dimensions) $\to$ Dropout.
- **Pre-LayerNorm Residual Formulation**:
  $$x^{(l)}_{1} = x^{(l-1)} + \text{Attention}(\text{LayerNorm}_1(x^{(l-1)}))$$
  $$x^{(l)} = x^{(l)}_{1} + \text{MLP}(\text{LayerNorm}_2(x^{(l)}_{1}))$$
  Pre-LN ensures stable gradient propagation directly through the residual stream without gradient vanishing/explosion.

### 5. Weight Tying & Weight Initialization
- **Weight Tying**: Reuses the token embedding table `wte.weight` as the projection matrix of `lm_head.weight` ($V \times C$), drastically reducing parameter count and improving semantic alignment.
- **Initialization**: Gaussian initialization $\mathcal{N}(0, 0.02)$ for all linear and embedding layers; biases initialized to zero.

---

## 📊 Training Dynamics & Results

The model was trained for **1,500 iterations** using **AdamW** ($\text{lr} = 10^{-3}$) on batch size $B=12$:

```
step    0: train loss 4.1746, val loss 4.1706  (initial uniform cross-entropy ~ ln(65) = 4.174)
step  150: train loss 2.5042, val loss 2.5067
step  300: train loss 2.4156, val loss 2.4430
step  450: train loss 2.2796, val loss 2.3042
step  600: train loss 2.1734, val loss 2.2065
step  750: train loss 2.1014, val loss 2.1520
step  900: train loss 2.0108, val loss 2.0994
step 1050: train loss 1.9498, val loss 2.0475
step 1200: train loss 1.9103, val loss 2.0077
step 1350: train loss 1.8443, val loss 1.9598
step 1500: train loss 1.8108, val loss 1.9714
```

| Metric | Starting Value (Step 0) | Final Value (Step 1500) |
| :--- | :--- | :--- |
| **Train Loss** | $4.1746$ | **$1.8108$** |
| **Validation Loss** | $4.1706$ | **$1.9714$** |
| **Perplexity** | $\approx 64.99$ | **$\approx 7.18$** |

---

## ✍️ Autoregressive Text Generation & Decoding Experiments

Autoregressive inference iteratively computes logits for the latest context position, modulates logits via **temperature scaling**, optionally restricts candidate tokens via **top-$k$ truncation**, applies **softmax**, and samples tokens via `torch.multinomial`:

$$\tilde{z}_i = \frac{z_i}{T}$$

### Temperature Scaling ($T=0.7$ vs $T=1.2$)
Prompt: `"KING:\n"` | Max tokens: 200 | Seed: 42

- **Low Temperature ($T = 0.7$)**:
  > *Sharpens the probability distribution, favoring confident token transitions. Yields structured words, realistic poetic cadence, and authentic Shakespearean formatting.*
  ```text
  KING:
  My God Will shall shall the like of thee the what wilst hese frease
  The miss my oughten sole, if worth is of are of lone
  Boome likes ef somall spale forn a her eards my clalke; would deake
  To read the
  ```

- **High Temperature ($T = 1.2$)**:
  > *Flattens the distribution, giving higher probability mass to tail characters. Boosts diversity and surprises, but produces more chaotic phrasing and non-word tokens.*
  ```text
  KING:
  My GOd Willksmage! Plast is lujedoughe.'

  OF what wilst Me;
  Then henne a pastmiood
  ThemeNout, if wo tearriounate of loyse of, Plitis;
  And yat subse arken a hence, do my visel .
  ```

---

### Top-$k$ Sampling (Full Vocab vs Top-10)
Prompt: `"KING:\n"` | Temperature: $1.0$ | Seed: 42

- **Unconstrained Sampling (`top_k = None`)**:
  > *Leaves the entire 65-token character set active. Long-tail improbable tokens occasionally get sampled, leading to minor spelling corruption.*
  ```text
  KING:
  My GOd Will sha? now know, Mlurdery thees hauld whese me.

  AUPEY:
  Thines if aminguess lestititire? I sair deat thoughm sow oman
  Theseef somal sirsed ratngell fleess in to alk.
  ```

- **Truncated Sampling (`top_k = 10`)**:
  > *Truncates choices strictly to the 10 highest-scoring character logits at each step, pruning noisy tail characters and generating coherent, fluent character strings.*
  ```text
  KING:
  My had subjeschannces, this duke of thees
  Of what wilst hese fressenned muss my oughtell,
  At, if wo that moselds of lone wo my likes;
  And yatusins harken a hen see simel wark.
  ```

---

## 📁 Repository Structure

```
├── nano_gpt_regressive_training.ipynb   # Complete implementation, training loop & sampling
├── input.txt                            # Tiny Shakespeare dataset corpus (1.11MB)
├── language_model_theory/               # Theoretical slides & lecture notes
│   └── theory_lecture.pdf
├── README.md                            # Project documentation & technical report
└── .gitignore                           # Git ignore rules
```

---

## 🚀 Quickstart & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Yashcode007/nano-gpt-autoregressive-lm.git
cd nano-gpt-autoregressive-lm
```

### 2. Set up Python environment
```bash
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install torch torchvision torchaudio matplotlib jupyter
```

### 4. Run the Jupyter Notebook
```bash
jupyter notebook nano_gpt_regressive_training.ipynb
```

---

## 📚 Citation & References

- **Vaswani et al. (2017)**: *Attention Is All You Need* ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762))
- **Radford et al. (2019)**: *Language Models are Unsupervised Multitask Learners* (GPT-2)
- **Karpathy, Andrej**: [nanoGPT](https://github.com/karpathy/nanoGPT)
- **Course**: *Advances in Deep Generative Modeling*, The University of Texas at Austin.

---

<p align="center">Made with ❤️ for Deep Generative Modeling Research & Exploration</p>
