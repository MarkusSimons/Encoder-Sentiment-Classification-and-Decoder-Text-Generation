# Encoder-Sentiment-Classification-and-Decode-Text-Generation

This repository contains my work for **Project 3** in **INF265 Deep Learning** at the University of Bergen. The project explores transformer-based architectures through two different tasks:

1. **Part I:** Sentiment classification on IMDb movie reviews using an encoder-only transformer
2. **Part II:** Question-answer generation / chatbot modeling using a decoder-only transformer

The project combines implementation work in PyTorch with training, evaluation, and analysis of transformer components such as self-attention, positional encoding, masking, and sequence modeling.

## Repository Structure

```text
project/
|-- encoder/
|   |-- imdb_sentiment_encoder.ipynb
|   |-- imdb_train/
|   `-- imdb_test/
|
`-- decoder/
    |-- chatbot.py
    |-- config.py
    |-- dataset.py
    |-- gpu_training_notebook.ipynb
    |-- inference.py
    |-- model.py
    |-- tokenizer.py
    |-- train.py
    |-- utils.py
    `-- gooaq_subset/
```

## Part I: Encoder Sentiment Classifier

Part I implements an encoder-based transformer for **binary sentiment classification** on IMDb movie reviews.

### Main elements

- Custom tokenization pipeline
- Use of special tokens such as `[CLS]` and `[PAD]`
- Positional encoding
- Multi-head self-attention
- Stacked encoder blocks with residual connections and layer normalization
- Binary classification using the final `[CLS]` representation


### Reported result

The encoder model achieved an accuracy of approximately **83.94%** on the sentiment classification task.

## Part II: Decoder Chatbot

Part II focuses on a **decoder-only transformer** for question-answer style text generation based on the `gooaq_subset` dataset.

### Main elements

- Byte-pair encoding (BPE) tokenizer training
- Decoder-style transformer architecture
- Causal masking for autoregressive generation
- Training with cross-entropy loss
- Inference with greedy and top-p sampling
- Optional Streamlit chatbot interface

### Current status

Part II includes the project template and training/inference scaffolding, but several `TODO` sections are still present in files such as `model.py`, `dataset.py`, and `inference.py`. In other words, this part is not yet fully complete in its current repository state.

## Requirements

This project uses Python together with common deep learning and NLP libraries, including:

- `torch`
- `datasets`
- `tokenizers`
- `numpy`
- `matplotlib`
- `tqdm`
- `scikit-learn`
- `streamlit`

If you want to install them manually, a typical setup would be:

```bash
pip install torch datasets tokenizers numpy matplotlib tqdm scikit-learn streamlit
```

## How to Run

### Part I

Open the notebook and run the cells:

```bash
jupyter notebook project_3_template/01_encoder_sentiment_classifier/imdb_sentiment_encoder.ipynb
```

### Part II

Train the tokenizer:

```bash
python project_3_template/02_decoder_chatbot/tokenizer.py
```

Train the decoder model:

```bash
python project_3_template/02_decoder_chatbot/train.py
```

Run local inference:

```bash
python project_3_template/02_decoder_chatbot/inference.py
```

Launch the chatbot interface:

```bash
streamlit run project_3_template/02_decoder_chatbot/chatbot.py
```

## Learning Goals

This project was mainly intended as hands-on practice with:

- implementing transformer components from scratch
- understanding sequence masking and attention
- working with tokenization and vocabulary design
- training neural sequence models in PyTorch
- analyzing model performance for both classification and generation tasks

## Notes
- Some datasets and intermediate files are stored locally inside the project folders.
- Part I is the most complete and directly reproducible section of the project in its current form.

## Authors:
Andreas Wold Sløgedal & Markus Dybevold Simonsen
