#  Email Spam Detection with Machine Learning using Python
**OIBSIP Task 4 – Data Science Internship Project**  
*Oasis Infobyte*

---

## Problem Statement

Spam or junk emails continue to flood inboxes with unsolicited, misleading, or malicious content. These emails often contain phishing links, scams, or irrelevant promotions. This project aims to create an automated spam detection system using Python and machine learning to classify emails as **spam** or **ham** (not spam).

---

## Objectives

1. **Data Preprocessing**  
   - Clean and normalize raw email text (lowercasing, punctuation removal).  
   - Drop duplicates and irrelevant columns.  
2. **Feature Engineering**  
   - Convert text into numerical features using **TF-IDF**.  
3. **Address Class Imbalance**  
   - Apply **SMOTE (Synthetic Minority Oversampling Technique)** to balance training data.  
4. **Model Training & Evaluation**  
   - Train a **Multinomial Naive Bayes** classifier.  
   - Evaluate using accuracy, precision, recall, F1-score, and ROC-AUC.  
5. **Deployment**  
   - Save the trained model and vectorizer using `joblib` for future prediction.

---

## Tools & Technologies

- **Language & IDE:** Python, Jupyter Notebook  
- **Libraries:** pandas, numpy, scikit-learn, imbalanced-learn (SMOTE), matplotlib, seaborn, wordcloud  
- **Modeling:** Multinomial Naive Bayes  
- **Saving/Deployment:** joblib  

---

##  Methodology

1. **Load & Clean Data**  
   - Read `spam.csv`, remove duplicate rows, rename columns, drop unused fields.  
   - Clean text: lowercase all messages, remove punctuation.  
2. **Explore Data**  
   - Inspect class distribution: 4,516 ham vs. 653 spam—class imbalance present.  
   - Visualize message length distributions and word clouds for both classes.  
3. **Vectorize Text**  
   - Convert cleaned messages into TF-IDF features.  
4. **Train/Test Split**  
   - Split data into training (80%) and testing (20%) sets with stratification.  
5. **Balance Training Data with SMOTE**  
   - Oversample the minority class to match majority class size (3,613 each).  
6. **Train Model**  
   - Fit a Multinomial Naive Bayes classifier on balanced data.  
7. **Evaluate Performance**  
   - Test on original (imbalanced) test set:  
     - **Accuracy:** 97.2%  
     - **Precision (spam):** 88.06%  
     - **Recall (spam):** 90.07%  
     - **F1-Score:** 89.05%  
     - **AUC Score:** 0.99  
   - Created confusion matrix and ROC curve.  
8. **Save Model & Vectorizer**  
   - Exported `.pkl` files for future inference.  

---

## Results Summary

| Metric      | Score   |
|-------------|---------|
| Accuracy    | 97.2%   |
| Precision   | 88.06%  |
| Recall      | 90.07%  |
| F1-Score    | 89.05%  |
| ROC-AUC     | 0.99    |

SMOTE significantly enhanced spam recall, ensuring fewer false negatives, making this model practical for real-world application.

---

---

##  Demo & LinkedIn

- **GitHub Repository:** https://github.com/YourUsername/OIBSIP_DataScience_Task4  
- **LinkedIn Post:** [See my project post](https://linkedin.com/in/yourprofile) 

---

##  Author

**Rajalakshmi V R**  
Data Science Intern – Oasis Infobyte | July 2025  
📧 rajalakshmirajan2021@gmail.com  
🔗 [linkedin Profile](https://linkedin.com/in/rajalakshmivr)

---

## Final Thoughts

This project showcases a real-world application of NLP and machine learning to solve a practical problem. By handling class imbalance via SMOTE and optimizing model performance, the resulting spam detector is both accurate and reliable. Further extensions could include deploying it as a web service or applying deep learning techniques for enhanced classification.

---

> Thank you, *Oasis Infobyte*, for enabling this hands-on learning experience!

