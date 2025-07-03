# 🧠 Youth Mental Health Narratives Analysis

This project uses real-world narrative and structured data from youth suicide cases to predict whether the individual had a recorded mental health problem.

## 📁 Dataset
- Format: Structured + unstructured (text)
- Fields: Case narratives (`NarrativeLE`, `NarrativeCME`) + behavioral flags (e.g., SuicideAttemptHistory, AlcoholProblem)
- Target variable: `MentalHealthProblem` (0 or 1)
- 📌 *Dataset not uploaded due to privacy — available upon request*

## 🔍 Objective
Use machine learning to classify whether a victim had a known mental health issue using both structured data and natural language narratives.

## 🧪 Tools & Technologies
- Python (pandas, scikit-learn, matplotlib, seaborn)
- Natural Language Processing (TF-IDF)
- Logistic Regression & Random Forest
- Cross-validation & GridSearchCV

## 🧠 Key Steps

1. Combined narrative fields into one text column
2. Extracted features using TF-IDF (5000 max features)
3. Selected relevant structured binary indicators
4. Combined both sets into one feature matrix
5. Trained models (Logistic Regression, Random Forest)
6. Evaluated using accuracy, classification report, confusion matrix
7. Tuned hyperparameters using GridSearchCV
8. Visualized feature importances

## ✅ Results

- **Random Forest Accuracy**: ~97.8%
- **Cross-validation accuracy**: ~98.0%
- **Top contributing features**: Narrative keywords, prior suicide attempts, family issues

## 📊 Sample Outputs

- Confusion Matrix
- Top 20 TF-IDF Features by Importance
- Correlation Heatmaps

## 📌 Conclusion

This project demonstrates how combining narrative text with structured features can improve predictive performance for mental health classification tasks. It highlights practical use of NLP, supervised learning, and feature importance analysis in real-world health data.
