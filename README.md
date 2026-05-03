# Intent & Sentiment-Aware Information Retrieval for Financial Q&A

> A hybrid retrieval pipeline that combines lexical search, neural re-ranking, and sentiment-driven diversity to help non-expert users navigate financial information safely.

---

## Overview

This project tackles a core challenge in financial information retrieval: non-expert users often encounter **one-sided advice**, **emotional bias**, and **vocabulary mismatch** when searching financial Q&A forums. Standard relevance-based systems surface the most popular answers — not necessarily the most balanced ones.

We designed and evaluated a modular IR pipeline on the [BEIR/FiQA](https://ir-datasets.com/beir.html#beir/fiqa) dataset (~57k documents, 1,296 queries from a financial community Q&A platform) that:

- **Distinguishes objective vs. subjective queries** via intent classification
- **Retrieves factually accurate documents** for factoid queries using neural re-ranking
- **Enforces sentiment diversity** for advice-seeking queries, deliberately surfacing contrasting viewpoints (buy/sell/neutral) to reduce echo chamber effects

**Central research question:** *Does intent-aware and sentiment-diverse retrieval improve ranked document diversity and retrieval quality for financial questions compared to standard relevance-based retrieval?*

---

## Pipeline Architecture

```
Documents & Queries
        │
        ▼
  Acronym Expansion + Preprocessing
        │
        ▼
    Indexing (PyTerrier)
        │
        ▼
   BM25 First-Stage Retrieval
        │
        ▼
   Doc2Query Augmentation
        │
        ▼
   Neural Re-Ranker (CrossEncoder)    ← Experiment 1 (E1)
        │
        ▼
   Intent Classifier
   ┌────┴────┐
Objective  Subjective
   │           │
No change  Sentiment Classifier (Pos / Neu / Neg)
               │
               ▼
        Diversity Re-Ranker  ← Experiment 2 (E2)
```

---

## Results

| System | MAP    | P@1    | P@5    | R@5    | nDCG@5  | nDCG@10 |
|--------|--------|--------|--------|--------|---------|---------|
| BM25 (baseline) | 0.2080 | 0.2330 | 0.1086 | 0.2490 | 0.2303  | 0.2571  |
| E1 – Neural Re-ranking | **0.2855** | **0.3441** | **0.1546** | **0.3525** | **0.3286** | **0.3477** |
| E2 – Sentiment Diversity | 0.2796 | 0.3364 | 0.1519 | 0.3455 | 0.3213  | 0.3426  |

**Key findings:**
- Neural re-ranking (E1) boosted MAP by ~**37%** and P@1 by ~**48%** over BM25, closing the vocabulary mismatch gap in the financial domain
- The slight metric drop in E2 is an intentional **"Diversity Tax"** — standard metrics penalise re-ordering even when it improves balance. Qualitatively, E2 transformed homogeneous result lists (e.g., five consecutive "Buy" recommendations) into heterogeneous ones interleaving opposing stances

---

## Tech Stack

| Component | Tool |
|-----------|------|
| IR Framework | [PyTerrier](https://github.com/terrier-org/pyterrier) |
| Dataset | [BEIR/FiQA](https://ir-datasets.com/beir.html#beir/fiqa) via `ir_datasets` |
| Document Expansion | [pyterrier_doc2query](https://github.com/terrierteam/pyterrier_doc2query) |
| Neural Re-ranking | `sentence-transformers` CrossEncoder |
| Intent & Sentiment | HuggingFace `transformers` pipelines |
| Query Expansion | WordNet (NLTK), SpaCy, custom financial glossary |
| Language | Python 3.10+ |

---

## Getting Started

### Prerequisites

```bash
pip install python-terrier langdetect tqdm sentence-transformers openpyxl
pip install -q transformers accelerate bitsandbytes
pip install --upgrade -q git+https://github.com/terrierteam/pyterrier_doc2query.git
```

### Dataset

The project uses the [FiQA](https://ir-datasets.com/beir.html#beir/fiqa) dataset, loaded automatically via `ir_datasets`:

```python
import pyterrier as pt
dataset = pt.get_dataset('irds:beir/fiqa/test')
```

### Running the Experiments

Open and run `Financial_Question_Answering.ipynb` in order. The notebook is self-contained and walks through:

1. Data loading & exploratory analysis
2. Baseline experiments (BM25, PL2, TF-IDF ± RM3)
3. Query expansion (semantic, morphological, acronym-based)
4. Document chunking experiments
5. Advanced pipeline: Doc2Query + CrossEncoder re-ranking (E1)
6. Intent + sentiment-aware diversity re-ranking (E2)

Pre-computed artefacts (expanded docs, rewritten queries) are fetched automatically from GitHub during setup.

---

## Team

This project was developed as part of the **Information Retrieval** course at [Università degli Studi di Milano-Bicocca](https://www.unimib.it/) (January 2026).

| Name | GitHub |
|------|--------|
| Michele Pio Lacagnina | [@MicheleLac](https://github.com/MicheleLac) |
| Linda Stalidzane | [@stalidzane](https://github.com/stalidzane) |
| Dario Zanini | — |

> Contributions were distributed across baseline experiments, advanced pipeline design, and analysis — see the [report](report/Project_Report.pdf) for a detailed breakdown.
