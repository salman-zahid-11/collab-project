📩 SMS Spam Classification using SVM
A machine learning project that classifies SMS messages as Spam or Ham (legitimate) using Support Vector Machines, with extensive feature engineering and model comparison.

📋 Table of Contents
Overview

Dataset

Features

Project Pipeline

Model Performance

Installation

Usage

Project Structure

Key Insights

Error Analysis

Saving & Loading the Model

Technologies Used

Author

🎯 Overview
This project builds a robust SMS spam classifier using Support Vector Machines (SVM). It goes beyond a simple TF-IDF + SVM baseline by:

Combining word-level and character-level TF-IDF features

Engineering hand-crafted meta features (digit counts, ALL-CAPS words, currency symbols, phone numbers, etc.)

Performing hyperparameter tuning with GridSearchCV

Comparing multiple classifiers (LinearSVC, Logistic Regression, ComplementNB)

Providing calibrated spam probabilities for threshold tuning

Including detailed error analysis and feature importance visualization

📊 Dataset
Source: SMS Spam Collection Dataset (Kaggle)

Total messages: 5,572

After removing duplicates: 5,169

Classes:

Ham: 4,825 (~86.6%)

Spam: 747 (~13.4%)

Format: CSV with v1 (label) and v2 (message) columns

The dataset is loaded as SMS.csv (encoded with latin-1).

🧠 Features
1. Text Preprocessing
Lowercasing

URL and email removal

Digit removal

Punctuation removal

Stopword removal

Lemmatization (WordNet)

2. TF-IDF Features
Type	Configuration
Word TF-IDF	ngram_range=(1,2), max_features=5000, sublinear_tf=True
Character TF-IDF	analyzer='char_wb', ngram_range=(2,5), max_features=10000
3. Hand-Crafted Meta Features (12 total)
Feature	Description
char_count	Number of characters
word_count	Number of words
avg_word_len	Average word length
digit_count	Number of digits
digit_ratio	Digits / total characters
upper_ratio	Uppercase letters / total characters
all_caps_words	Count of ALL-CAPS words
exclaim_count	Number of !
special_count	Number of special characters
has_url	Contains a URL (binary)
has_currency	Contains £, $, or € (binary)
has_phone	Contains a 5+ digit sequence (binary)
Meta features are log-scaled (np.log1p) and scaled with MaxAbsScaler.

🔄 Project Pipeline
text
┌─────────────────────┐
│  Load SMS.csv       │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  EDA & Cleaning     │  (word clouds, class distribution, duplicates)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Text Preprocessing │  (clean_text function)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Feature Extraction │  (word TF-IDF + char TF-IDF + meta features)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Train/Test Split   │  (80/20, stratified)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Baseline SVM       │  (LinearSVC + word TF-IDF)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Hyperparameter     │  (GridSearchCV over C & class_weight)
│  Tuning             │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Enhanced Model     │  (word + char + meta features)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Model Comparison   │  (SVM vs LR vs ComplementNB)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Evaluation         │  (Confusion Matrix, ROC, PR curves)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Error Analysis &   │
│  Feature Importance │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Calibrated Model   │  (spam probability output)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Save with joblib   │
└─────────────────────┘
📈 Model Performance
Baseline vs Enhanced (on held-out test set)
Model	Accuracy	Precision	Recall	F1
Baseline (word TF-IDF)	0.9845	0.9832	0.8931	0.9360
Enhanced (word + char + meta)	0.9903	0.9840	0.9389	0.9609
5-Fold Cross-Validation Comparison
Model	Accuracy	Precision	Recall	F1
LinearSVC	0.9894	0.9709	0.9444	0.9573
LogisticRegression	0.9889	0.9671	0.9444	0.9554
ComplementNB	0.9874	0.9683	0.9310	0.9492
Calibrated Model
ROC-AUC: 0.9971

Best threshold (F1): 0.3 → F1 = 0.9650

Threshold Tuning Table
Threshold	Precision	Recall	F1
0.3	0.9841	0.9466	0.9650
0.4	0.9840	0.9389	0.9609
0.5	1.0000	0.9313	0.9644
0.6	1.0000	0.9237	0.9603
0.7	1.0000	0.9084	0.9520
0.8	1.0000	0.8931	0.9435
⚙️ Installation
Requirements
bash
pip install pandas numpy matplotlib seaborn wordcloud nltk scikit-learn joblib
NLTK Data
python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
🚀 Usage
1. Prepare the Dataset
Download SMS.csv from Kaggle and place it in the same directory as the notebook.

2. Run the Notebook
Open SMS_Spam_SVM_By_SALMAN.ipynb in Google Colab or Jupyter and run all cells sequentially.

3. Predict a Single Message
python
predict_sms("URGENT! You've won a £1000 prize. Call 09061701461 now!")
Output:

text
Spam  (spam probability: 99.9%)
Signals: mentions money, contains a phone number / short code, 1 ALL-CAPS word(s), 1 exclamation mark(s)
Message: URGENT! You've won a £1000 prize. Call 09061701461 now!
4. Batch Prediction
python
messages = [
    "Congratulations! You have won a free iPhone.",
    "Hi, are we meeting tomorrow at 3 PM?",
    "FREE entry in a weekly competition! Text WIN to 80086 now!",
]
results = predict_many(messages)
print(results)
5. Load the Saved Model
python
import joblib
model = joblib.load('sms_spam_svm.joblib')

# Note: clean_text and extract_meta_features must be defined
prediction = model.predict(pd.DataFrame({
    'message': ["Free entry to win!"],
    'clean_message': [clean_text("Free entry to win!")]
}))
📁 Project Structure
text
.
├── SMS_Spam_SVM_By_SALMAN.ipynb   # Main notebook
├── SMS.csv                        # Dataset (downloaded from Kaggle)
├── sms_spam_svm.joblib            # Saved model pipeline
└── README.md                      # This file
💡 Key Insights
Character-level TF-IDF captures spam signals that word-level TF-IDF misses (e.g., £, !!!, www, txt).

Meta features add strong discriminative power — spam messages average 15 digits vs. 0.3 for ham.

has_phone is highly predictive: 76.6% of spam contains phone-like sequences vs. 0.1% of ham.

Class imbalance matters — using class_weight='balanced' improves recall on the minority (spam) class.

F1 is a better metric than accuracy for this imbalanced problem.

Calibrated probabilities allow threshold tuning — a threshold of 0.3 maximizes F1.

🔍 Error Analysis
False Positives (Ham flagged as Spam)
Legitimate messages that mention money or prices can be misclassified:

"Hey...Great deal...Farm tour 9am to 5pm $95/pax, $50 deposit by 16 May"

"Your bill at 3 is å£33.65 so thats not bad!"

False Negatives (Spam Missed)
Spam that mimics ham-style language or lacks typical signals:

"Latest News! Police station toilet stolen, cops have nothing to go on!"

"Would you like to see my XXX pics they are so hot..."

💾 Saving & Loading the Model
The full pipeline (vectorizers + scaler + calibrated SVM) is saved using joblib:

python
import joblib
joblib.dump(final_model, 'sms_spam_svm.joblib')
⚠️ Important: When loading the model, ensure clean_text and extract_meta_features are defined in your environment, as the pipeline depends on them.

🛠️ Technologies Used
Category	Tools
Language	Python 3
Data Processing	pandas, numpy
NLP	NLTK (stopwords, WordNet lemmatizer)
Feature Extraction	scikit-learn (TfidfVectorizer, ColumnTransformer)
Modeling	scikit-learn (LinearSVC, LogisticRegression, ComplementNB, CalibratedClassifierCV)
Hyperparameter Tuning	GridSearchCV, StratifiedKFold
Visualization	matplotlib, seaborn, wordcloud
Model Persistence	joblib
Environment	Google Colab
👤 Author
Salman

Notebook: SMS_Spam_SVM_By_SALMAN.ipynb

Dataset: UCI SMS Spam Collection

📜 License
This project is intended for educational purposes. The dataset is provided by UCI Machine Learning Repository via Kaggle.

⭐ Acknowledgements
UCI Machine Learning Repository for the SMS Spam Collection dataset

scikit-learn for the ML toolkit

NLTK for NLP utilities

Google Colab for the free GPU/CPU environment
