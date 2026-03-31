# Toxic Comment Classification (Multi-Label NLP)

> Multi-label toxicity detection using classical machine learning, CNNs, and ensemble methods under severe class imbalance.

---

## Overview
This project studies multi-label toxicity detection on Wikipedia talk pages using the Jigsaw Toxic Comment dataset.  
The objective is to identify multiple categories of harmful content simultaneously, including:

- toxic  
- severe_toxic  
- obscene  
- threat  
- insult  
- identity_hate  

The project compares classical machine learning models, deep learning architectures, and ensemble methods to understand trade-offs between performance, robustness, and interpretability.

---

## Key Results
- Logistic Regression (TF-IDF): **ROC-AUC 0.978**  
- 1D CNN: **ROC-AUC 0.978**  
- Ensemble (LogReg + CNN): **ROC-AUC 0.9816** *(best performance)*  
- Improved rare-class detection via **threshold optimization**

---

## Key Contributions
- Built a complete **multi-label classification pipeline**  
- Conducted **exploratory data analysis** (class imbalance, co-occurrence, text patterns)  
- Designed **dual preprocessing pipelines**:
  - TF-IDF for classical models  
  - Tokenized sequences for neural models  
- Implemented and compared:
  - Logistic Regression, Linear SVM  
  - Random Forest, LightGBM  
  - 1D Convolutional Neural Network  
- Developed a **heterogeneous ensemble model**  
- Applied **threshold tuning** to improve performance on rare classes  

---

## Methods
- TF-IDF + Logistic Regression  
- Linear SVM  
- Random Forest & LightGBM  
- 1D CNN (embedding + convolution + pooling)  
- Ensemble learning (probability averaging)

---

## Project Structure

toxic-comment-classification/
├── notebooks/ # experiments and full pipeline
├── src/ # preprocessing, models, training scripts
├── figures/ # visualizations
├── report/ # full project report (PDF)
├── data/ # dataset instructions (no raw data included)
├── README.md
└── requirements.txt



---

## Tech Stack
Python, NumPy, Pandas, scikit-learn, TensorFlow/Keras, LightGBM, Matplotlib

---

## Data
This project uses the **Jigsaw Toxic Comment Classification dataset**:  
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge  

Download the dataset and place it in the `data/` directory before running the code.

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
2. Download dataset from Kaggle
3. Run:
    jupyter notebook notebooks/G9_Toxic_comments.ipynb

Notes
Accuracy is not used due to class imbalance
Evaluation is based on ROC-AUC and per-label metrics
Threshold tuning is critical for detecting rare classes


Full methodology and results are available in:
report/toxic_comments_report.pdf