# SMS Spam Detector

This project trains a machine learning model to classify SMS messages as ham or spam.

## Features
- Loads the SMS Spam Collection dataset
- Uses TF-IDF vectorization + Multinomial Naive Bayes
- Splits data into train/test sets with stratification
- Evaluates model accuracy and classification metrics
- Tests the model on sample messages

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and place it in the project folder as `spam.csv`.

   You can get it from:
   - UCI SMS Spam Collection dataset
   - Kaggle "spam.csv" dataset

4. Run the project:
   ```bash
   python sms_spam_detector.py
   ```

## Dataset format
The script expects one of these column layouts:
- `v1` = label, `v2` = message
- or `label` and `message`

The labels are mapped as:
- `ham` = 0
- `spam` = 1

## Notes
This is a simple baseline model. It works well for SMS filtering and is a good starting point for further improvements such as:
- custom text preprocessing
- better class balancing
- model comparison (SVM, Logistic Regression, XGBoost)
