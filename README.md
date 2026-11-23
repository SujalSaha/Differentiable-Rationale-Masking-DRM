# Differentiable Rationale Masking (DRM) for Hallucination Elimination in LLMs

An end-to-end, trainable rationale-routing architecture that eliminates Large Language Model (LLM) hallucinations by enforcing causal evidence attribution at middle Transformer Layer $L$.

---

## Project Overview

Standard Retrieval-Augmented Generation (RAG) and long-context LLMs often ignore retrieved passages or blend parametric memory with noisy context, producing ungrounded claims. **Differentiable Rationale Masking (DRM)** inserts a lightweight, trainable **Sentence Highlighter Switch** (Gumbel-Sigmoid router) at middle layer $L$ of a causal Transformer:

1. **Lower Layers ($1 \dots L-1$) — Isolation:** Document and answer representations are built independently via a structural isolation mask to prevent premature cross-attention leakage.
2. **Middle Layer ($L$) — Differentiable Routing:** Document tokens are mean-pooled per sentence and scored against `[route]` clause trigger tokens using a Straight-Through Gumbel-Sigmoid switch ($M \in \{0, 1\}^K$).
3. **Upper Layers ($L \dots L_{\text{max}}$) — Hard Attention Suppression:** Sentences switched OFF (`0`) receive a large negative attention bias ($-C_{\text{pen}}$), physically preventing upper layers from reading unselected context.
4. **3-Way "Prove-It" Contrastive Training:** Trains the router via **Normal ($M$)**, **Flipped ($1 - \tilde{M}$)**, and **Empty ($0$)** forward passes alongside a **Bernoulli KL-Divergence sparsity penalty** (target $\rho = 0.08$).

---

## Repository Structure

```text
Differentiable-Rationale-Masking-DRM/
├── Differentiable_Rationale_Masking_DRM.ipynb   # Unified 4-phase Google Colab notebook with verified outputs
└── README.md                                    # Project documentation, architecture, and setup guide
