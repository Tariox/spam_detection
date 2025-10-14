# SMS Spam Detection using Machine Learning

This project builds an **AI model to classify SMS messages** as **Spam** or **Legitimate (Ham)** using **Natural Language Processing (NLP)** techniques such as **TF-IDF** vectorization and **Machine Learning** classifiers like **Naive Bayes**, **Logistic Regression**, and **Support Vector Machine (SVM)**.

---

## Project Overview

With the rapid increase of unwanted promotional and fraudulent text messages, it is essential to automatically detect and filter out spam messages.  
This model analyzes message content and predicts whether it is spam or not based on patterns learned from historical data.

---

## Features

- Preprocesses text data (cleaning, tokenization, and stopword removal)  
- Uses **TF-IDF Vectorization** to convert text into numerical features  
- Trains multiple ML models:
  - **Naive Bayes**
  - **Logistic Regression**
  - **Support Vector Machine (SVM)**
- Evaluates model performance using:
  - Accuracy Score
  - Classification Report
  - Confusion Matrix (visualized with Seaborn)
- Predicts new SMS messages as **Spam** or **Legit**

---

## Technologies Used

| Category | Tools / Libraries |
|----------|------------------|
| Programming Language | Python |
| Data Handling | Pandas, NumPy |
| NLP | Scikit-learn (TF-IDF) |
| Machine Learning | Naive Bayes, Logistic Regression, SVM |
| Visualization | Matplotlib, Seaborn |

---

## Dataset

**Dataset Name:** SMS Spam Collection Dataset  
**Source:** [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

Contains 5,574 messages labeled as:
- **ham** → legitimate message  
- **spam** → unwanted promotional or fraudulent message

---
### Clone the repository
```bash
git clone https://github.com/<your-username>/sms_spam_detection.git
cd sms_spam_detection
```
### Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Mac/Linux
```
### Install dependencies
```bash
pip install -r requirements.txt
```
### Watch demon
Spam_detection - [Watch Demo](https://drive.google.com/file/d/1hXWz2K252erLgLXlrzD09J7F6Mazaosl/view?usp=sharing)
