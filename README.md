# Twitter Sentiment Classification (4-Class)

A machine learning group project that classifies tweets referencing companies and entities into four sentiment classes — **Positive**, **Negative**, **Neutral**, and **Irrelevant** — comparing three classical machine learning models against a Bidirectional GRU deep learning model.

## Overview

This project builds and compares two families of text classification approaches on the same dataset:

- **Classical ML** (TF-IDF features): Multinomial Naive Bayes, Logistic Regression, Linear SVM
- **Deep Learning** (learned word embeddings + sequence modeling): Bidirectional GRU with an Embedding layer

Both approaches are evaluated on identical metrics (accuracy, macro F1, per-class precision/recall) to give an honest, evidence-based comparison rather than assuming one approach is automatically better.

## Dataset

- **Source**: [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) (Kaggle)
- **Files**: `twitter_training.csv`, `twitter_validation.csv`
- **Columns**: `id`, `company`, `sentiment`, `raw_tweet`
- **Classes**: Positive, Negative, Neutral, Irrelevant

Download the dataset from Kaggle and place both CSVs in a `tweets/` folder at the project root:

```
project-root/
├── tweets/
│   ├── twitter_training.csv
│   └── twitter_validation.csv
├── Twitter_Sentiment.ipynb
├── README.md
└── requirements.txt
```

## Pipeline

1. **Load & clean** — read both CSVs (handles malformed rows/quotes), drop nulls and duplicates
2. **EDA** — class distribution bar charts, word clouds per sentiment class
3. **Preprocessing** — lowercase, strip non-letters, remove stopwords, stem (NLTK)
4. **Label encoding** — `LabelEncoder` for the 4 sentiment classes
5. **Feature extraction**:
   - TF-IDF (unigrams + bigrams, 5,000 features) for the classical models
   - Tokenized, padded integer sequences (max length 40) for the GRU
6. **Model training** — 3 classical models + 1 Bidirectional GRU, all on the same underlying cleaned text
7. **Evaluation** — accuracy, macro F1, per-class classification report, confusion matrices for all 4 models
8. **Feature importance** — top TF-IDF words per class (Logistic Regression coefficients)
9. **Sample predictions** — spot-check on real validation tweets + a reusable `predict_sentiment()` function for live demos on new text
10. **Model saving** — best model (by macro F1) and its preprocessing artifacts saved to disk

## Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK stopwords (one-time)
python -c "import nltk; nltk.download('stopwords')"
```

### requirements.txt
```
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
wordcloud
tensorflow
joblib
```

> **Note on TensorFlow/GRU training**: training the Bidirectional GRU is significantly faster with a GPU. If you don't have one locally, run the notebook in [Google Colab](https://colab.research.google.com) (free GPU runtime) rather than on CPU.

## Running the Project

Open `Twitter_Sentiment.ipynb` in Jupyter, VS Code, or Colab and run all cells top to bottom. Each section is documented with markdown explaining what it does and why.

## Results

| Model | Type | Accuracy | Macro F1 |
|---|---|---|---|
| Multinomial Naive Bayes | Classical (TF-IDF) | 66.16% | 0.6498 |
| Logistic Regression | Classical (TF-IDF) |75.79% | 0.7531 |
| Linear SVM (OvR) | Classical (TF-IDF) | 78.93% |0.7840 |
| Bidirectional GRU + Embedding | Deep Learning |88.86% | 0.8895 |

See the full project report for detailed methodology, per-class results, and discussion of the classical vs. deep learning trade-offs.

## Known Issues Encountered & Fixed

- **CSV parsing error** (`EOF inside string`): some tweet text contains unclosed quote characters, which breaks pandas' default C parser. Fixed by using `engine='python'` with `quoting=csv.QUOTE_NONE`.
- **Preprocessing bug**: an earlier version of the cleaning function computed cleaned tokens correctly but returned the original unprocessed text. Fixed and verified with a before/after sanity check in the notebook.

## Project Structure

```
├── tweets/                                    
├── Twitter_Sentiment.ipynb     
├── README.md
└── requirements.txt
```

## Group Members

| Name | Index Number | GitHub Repository |
|---|---|---|
| EMMANUEL BAIDOO | UEB3519423 | https://github.com/HUMBLE-DEV/Tweeter-X-Sentiment-Analysis |
| KINGSLEY BOAMAH OWUSU  | UEB3522223 | https://github.com/KingsleyOwusuBoamah/Twetter_Sentiment_Analysis | 
| ISAAC HWEDIE OSEI | UEB3508223  | https://github.com/oseihwedieisaac/Twitter_Sentiment_Analysis_X |

## License

Educational project — University of Energy and Natural Resources (UENR), Department of Computer Science, Artificial Intelligence (Machine Learning) coursework.
