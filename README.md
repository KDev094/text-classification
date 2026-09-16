# Text Classification: IMDB Sentiment Analysis

This project explores binary sentiment classification for IMDB movie reviews.  Each review is classified as either **negative** or **positive** through three complementary notebook-based experiments:

- traditional machine-learning models over TF-IDF features;
- a compact PyTorch `EmbeddingBag` classifier; and
- bidirectional LSTM models initialized with GloVe word vectors.

The notebooks are designed as an educational comparison of feature-based and neural approaches, rather than as a packaged training application.

## Notebooks

| Notebook | Focus | Main approach |
| --- | --- | --- |
| `text_classification_sklearn.ipynb` | Classical baseline comparison | TF-IDF (up to 10,000 features) with Logistic Regression, MLP, Linear SVC, ensemble methods, and Naive Bayes |
| `text_classification_pytorch_v1.ipynb` | Lightweight neural baseline | `EmbeddingBag` + linear layer, trained with cross-entropy loss |
| `text_classification.ipynb` | Sequence-model experiments | Static 300-dimensional GloVe embeddings with three BiLSTM variants; the final variant uses two-class log-softmax output and early stopping |

## Data

The notebooks load the [IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) through Hugging Face `datasets`. It contains 25,000 labeled training reviews and 25,000 labeled test reviews.

`text_classification.ipynb` combines and re-splits the corpus with stratification, then uses a 90/10 training/validation split. Its prepared splits are included in this repository:

| File | Rows | Columns |
| --- | ---: | --- |
| `train.csv` | 36,000 | `text`, `label`, `tokenized_text`, `text_len` |
| `valid.csv` | 4,000 | `text`, `label`, `tokenized_text`, `text_len` |
| `test.csv` | 10,000 | `text`, `label`, `tokenized_text`, `text_len` |

The scikit-learn and `pytorch_v1` notebooks use the dataset's original train/test partitions and remap labels from `{0, 1}` to `{1, 2}` before their respective model pipelines.

## Preprocessing

Across the notebooks, the review-cleaning workflow includes the following operations (the exact sequence varies slightly by notebook):

- expand common contractions and correct selected malformed or concatenated tokens;
- remove HTML with Beautiful Soup;
- remove URLs, emojis, symbols, and punctuation where applicable;
- lowercase and lemmatize text with NLTK; and
- tokenize with TorchText's `basic_english` tokenizer for the PyTorch workflows.

For the BiLSTM workflow, the vocabulary is built from the training split with a maximum of 50,000 tokens. Reviews are represented as token IDs, and the sequence length is chosen from the 90th percentile of training-review lengths (the recorded run uses 200 tokens).

## Reported notebook results

The figures below are outputs saved in the notebooks. They are useful reference points, but will vary with package versions, hardware, randomization, and preprocessing changes.

| Experiment | Test accuracy |
| --- | ---: |
| TF-IDF + Logistic Regression | 0.88 |
| TF-IDF + MLPClassifier | 0.88 |
| TF-IDF + LinearSVC | 0.87 |
| PyTorch `EmbeddingBag` baseline | 0.862 |
| GloVe BiLSTM (best saved checkpoint) | 0.863 |

The BiLSTM checkpoint is selected by validation loss with patience-based early stopping. In the recorded run, the best validation accuracy was 0.868 and the final loaded checkpoint achieved 0.863 test accuracy.

## Setup

Use Python 3.9 or newer in a virtual environment, then install the notebook dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyterlab torch torchtext torchinfo datasets transformers \\
  scikit-learn pandas numpy matplotlib nltk beautifulsoup4 xgboost
```

Download the NLTK data used for tokenization, stopword filtering, POS tagging, and lemmatization:

```bash
python -c "import nltk; nltk.download('popular')"
```

Start Jupyter and open a notebook:

```bash
jupyter lab
```

Run notebook cells from top to bottom. The first use of `datasets.load_dataset('imdb')` downloads the IMDB data if it is not already cached locally.

## Running the experiments

1. Start with `text_classification_sklearn.ipynb` for a fast, interpretable TF-IDF baseline.
2. Run `text_classification_pytorch_v1.ipynb` to compare a simple learned embedding model.
3. Run `text_classification.ipynb` for the full preprocessing pipeline and GloVe/BiLSTM variants. It can reuse the included CSV files after they have been prepared.

The BiLSTM notebook downloads `GloVe.6B` 300-dimensional embeddings through TorchText. Its current cache path is hard-coded as `/workspace/books/.vector_cache`; update that path to a writable directory on your machine before running the GloVe cell. Training will use CUDA automatically when it is available.

## Repository layout

```text
.
├── text_classification_sklearn.ipynb      # TF-IDF and classical-model benchmarks
├── text_classification_pytorch_v1.ipynb   # EmbeddingBag neural baseline
├── text_classification.ipynb              # GloVe BiLSTM experiments
├── train.csv                              # Prepared training data
├── valid.csv                              # Prepared validation data
└── test.csv                               # Prepared test data
```

## Notes

- These are exploratory notebooks. They do not pin dependency versions or provide a command-line training interface.
- `text_classification.ipynb` saves its best final-model weights as `model-3.pth` in the working directory; the file is not included in this repository.
- Some imported libraries are notebook-specific or exploratory (for example, `xgboost` is imported in the scikit-learn notebook even though its corresponding experiment cell instantiates a random forest).

