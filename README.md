# Neural Sentiment Classifier

Binary sentiment classification on tweets, comparing four neural network architectures — FFNN, RNN, LSTM and GRU — trained under identical hyperparameters so the comparison is fair.

University project, Faculty of Computers and Artificial Intelligence, Sphinx University.

## Team

- [Your name]
- Ali Sayed
- Mohammed Hazem
- Mohamed Hgagy
- Ahmed Adel

## Overview

We built a sentiment classifier for short, informal text (tweets) and used it as a testbed to compare how a feedforward network, a vanilla RNN, an LSTM and a GRU handle the same task under the same training budget. Rather than picking one architecture up front, we trained all four with matching hidden sizes, dropout, optimizer settings and early stopping criteria, then evaluated them on a held-out test set to see which one actually earns its extra complexity.

## Dataset

We used the `twitter_samples` corpus from NLTK: 5,000 positive and 5,000 negative tweets, perfectly balanced. It's a small, clean, well-known benchmark (the same corpus used in several NLP courses), which makes it easy to work with but also means it isn't fully representative of noisy, real-world Twitter data — see Limitations below.

## Preprocessing

Tweets are cleaned with NLTK's `TweetTokenizer` (handles stripped, elongated words normalized) after removing URLs, HTML entities and hashtag symbols, plus a pass that drops punctuation-only tokens.

We also mark negation scope: for four tokens following a negation word (`not`, `never`, `isn't`, ...), the token is flagged as negated. Initially this was done by literally renaming the token (`bad` → `NOT_bad`), but that broke the pretrained embedding lookup, since compound tokens like `NOT_bad` don't exist in GloVe's vocabulary — they just collapsed to a zero vector, silently discarding the word's meaning. We fixed this by embedding the base word (`bad`) normally and appending a separate binary dimension marking negation. That change alone dropped the average out-of-vocabulary rate per tweet from roughly 1.57 (negative) / 1.19 (positive) down to 0.67 / 0.79.

Each token is represented by its 100-dimensional GloVe Twitter embedding plus that negation flag (101 dimensions total), padded or truncated to 30 tokens per tweet.

## Models

- **FFNN** — masked mean-pooling over the token embeddings, followed by two dense layers. No sequence modeling at all; the baseline for whether recurrence is worth it here.
- **RNN** — a vanilla bidirectional, 2-layer RNN.
- **LSTM** — bidirectional, 2-layer LSTM.
- **GRU** — bidirectional, 2-layer GRU.

All three recurrent models use `pack_padded_sequence` so padding tokens are never processed as real timesteps, and read out the concatenated final forward/backward hidden state into a single classification head.

## Training setup

Shared across all four models: hidden size 128, dropout 0.2, Adam (lr 1e-3, weight decay 1e-4), batch size 64, gradient clipping at norm 5.0, up to 50 epochs with early stopping on validation loss (patience 6, best-epoch weights restored). Data is split 70/15/15 into train/validation/test, stratified by label, with a fixed random seed — the test set is only touched once, for final evaluation.

## Results

| Model | Params | Train time | Best epoch | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| FFNN | 26,369 | 16.1s | 12 | 0.768 | 0.772 | 0.760 | 0.766 |
| RNN | 158,209 | 25.9s | 2 | 0.775 | 0.772 | 0.780 | 0.776 |
| LSTM | 632,065 | 90.4s | 3 | 0.783 | 0.787 | 0.776 | 0.781 |
| GRU | 474,113 | 78.8s | 3 | 0.785 | 0.795 | 0.769 | 0.782 |

GRU comes out slightly ahead on accuracy and F1, with LSTM close behind. The RNN stops improving after just 2 epochs, which is consistent with vanilla RNNs being less stable on longer sequences even with gradient clipping. The FFNN baseline is only a few points behind the recurrent models — tweets are short enough that pooled embeddings alone go a long way, which is worth keeping in mind before reaching for a heavier model.

Full training curves, a confusion matrix per model, and a per-metric comparison chart are in the notebook.

## Known limitation

The negation-flag fix stops words from losing their embedding, but it doesn't teach the model deep compositional semantics on its own. Short phrases like "not bad actually" are still sometimes misclassified — the model picks up on the negation cue but doesn't always resolve the resulting polarity flip correctly. This is a reasonable next thing to dig into rather than something we consider solved.

## Setup

```
pip install -r requirements.txt
```

The first run downloads the NLTK `twitter_samples` corpus and the `glove-twitter-100` embeddings (roughly 400MB) via `gensim`; both are cached locally afterward. On CPU, a full run of the notebook (data loading, all four models, evaluation) takes about 10–15 minutes, most of it spent loading the GloVe vectors.

## Repository structure

```
neural_sentiment_classifier.ipynb   main notebook: EDA, preprocessing, models, training, evaluation
requirements.txt                    Python dependencies
README.md
```

## Future work

- Test generalization on a larger, noisier dataset (e.g. Sentiment140) instead of this curated, balanced corpus.
- Try fine-tuning the embeddings instead of keeping them fixed.
- Add a small attention layer for interpretability.
- Wrap `predict_sentiment` in a lightweight Streamlit or Gradio demo.
