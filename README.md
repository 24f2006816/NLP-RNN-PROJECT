# Simple RNN — IMDB Sentiment Analysis

A Natural Language Processing project that uses a **Simple Recurrent Neural Network (RNN)** to perform binary sentiment classification on IMDB movie reviews.

The project covers the complete workflow from text preprocessing and word embeddings to RNN model training and sentiment prediction.

---

## Project Overview

Sentiment analysis is a Natural Language Processing task used to determine the emotional polarity of a piece of text.

In this project, an RNN is trained on the **IMDB Movie Reviews dataset** to classify reviews into two categories:

* **Positive**
* **Negative**

The project demonstrates how sequential text data can be processed and learned using recurrent neural networks.

---

## Project Workflow

```text
IMDB Reviews
      │
      ▼
Text Preprocessing
      │
      ▼
Tokenization
      │
      ▼
Word Embedding
      │
      ▼
Sequence Representation
      │
      ▼
Simple RNN
      │
      ▼
Dense Output Layer
      │
      ▼
Sentiment Prediction
```

---

## Tech Stack

* **Python**
* **TensorFlow / Keras**
* **NumPy**
* **Pandas**
* **NLP**
* **Simple RNN**
* **Word Embeddings**
* **Jupyter Notebook**

---

## Project Structure

```text
NLP-RNN-PROJECT/
│
├── embedding.ipynb
├── simplernn.ipynb
├── prediction.ipynb
├── main.py
├── requirements.txt
├── simple_rnn_imdb.h5
└── README.md
```

### File Description

| File                 | Description                                                    |
| -------------------- | -------------------------------------------------------------- |
| `embedding.ipynb`    | Exploratory work related to text representation and embeddings |
| `simplernn.ipynb`    | Training and evaluation of the Simple RNN model                |
| `prediction.ipynb`   | Testing the trained model on new reviews                       |
| `main.py`            | Python script for running the prediction workflow              |
| `requirements.txt`   | Required Python dependencies                                   |
| `simple_rnn_imdb.h5` | Trained RNN model                                              |

---

## Model Architecture

The project uses a Simple RNN-based architecture for sequence classification.

```text
Input Text
    │
    ▼
Tokenization
    │
    ▼
Embedding Layer
    │
    ▼
Simple RNN Layer
    │
    ▼
Dense Layer
    │
    ▼
Sigmoid Output
    │
    ▼
Positive / Negative
```

The **Embedding Layer** converts tokenized words into dense vector representations, while the **Simple RNN** processes the sequence and learns contextual patterns from the review.

The final output uses a sigmoid-based binary classification approach.

---

## Dataset

The project uses the **IMDB Movie Reviews dataset**, a widely used benchmark dataset for binary sentiment classification.

Each review is associated with one of two labels:

```text
0 → Negative
1 → Positive
```

The model learns patterns in the text that help distinguish positive reviews from negative reviews.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/24f2006816/NLP-RNN-PROJECT.git
```

Move into the project directory:

```bash
cd NLP-RNN-PROJECT
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Using Jupyter Notebook

Start Jupyter:

```bash
jupyter notebook
```

Then run the notebooks in the following order:

```text
1. embedding.ipynb
2. simplernn.ipynb
3. prediction.ipynb
```

---

### Using Python

The prediction workflow can also be executed using:

```bash
python main.py
```

---

## Example

The trained model can be used to classify a new movie review.

### Positive Review

```text
"The movie was absolutely amazing. The story was engaging
and the performances were excellent."
```

Expected sentiment:

```text
Positive
```

### Negative Review

```text
"The movie was boring and poorly written.
I did not enjoy watching it."
```

Expected sentiment:

```text
Negative
```

---

## Key Concepts Covered

This project provides practical implementation of several NLP and Deep Learning concepts:

* Natural Language Processing
* Text preprocessing
* Tokenization
* Vocabulary creation
* Sequence representation
* Word embeddings
* Recurrent Neural Networks
* Simple RNN architecture
* Binary classification
* Sentiment analysis
* Model training
* Model evaluation
* Model inference

---

## Learning Objectives

Through this project, the following concepts are explored:

1. Understanding how text can be converted into numerical representations.
2. Understanding word embeddings.
3. Working with sequential text data.
4. Understanding the basic architecture of RNNs.
5. Training an RNN for binary classification.
6. Using a trained model for real-world text prediction.
7. Building an end-to-end NLP deep learning pipeline.

---

## Future Improvements

The project can be extended with more advanced NLP architectures such as:

* LSTM
* GRU
* Bidirectional RNN
* Bidirectional LSTM
* Attention Mechanisms
* Transformer-based models
* BERT
* DistilBERT

Additional improvements could include:

* Hyperparameter tuning
* Better text preprocessing
* Model performance visualization
* REST API deployment
* Streamlit web application
* Dockerization
* Cloud deployment

---

## Author

**Pratyaksh Pandey**

IIT Madras — BS in Data Science and Applications

GitHub: [24f2006816](https://github.com/24f2006816)

---

## License

This project is intended for educational and learning purposes.
