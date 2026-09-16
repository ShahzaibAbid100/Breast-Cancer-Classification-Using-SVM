# Breast Cancer Classification Using SVM

##  Project Overview

This project implements a **Support Vector Machine (SVM)** classification model to classify breast cancer cases as **Benign (B)** or **Malignant (M)** using numerical features from the Breast Cancer Wisconsin dataset.

The project demonstrates a complete machine learning workflow including data preprocessing, feature scaling, SVM model training, cross-validation, kernel comparison, hyperparameter tuning, and model evaluation.

---

##  Project Objective

The objective of this project is to build an SVM classification model that predicts whether a tumor is:

- **B** → Benign
- **M** → Malignant

---

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset**.

The dataset contains multiple numerical features describing characteristics of cell nuclei, including measurements related to:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Concave points
- Fractal dimension

The target column is:

```text
diagnosis
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab / Jupyter Notebook

---

## 🔄 Project Workflow

1. Import required libraries
2. Load the dataset
3. Explore the dataset
4. Check missing values
5. Check and remove duplicates if necessary
6. Remove unnecessary columns
7. Analyze the target variable
8. Separate features and target
9. Split data into training and testing sets
10. Apply feature scaling
11. Train a baseline SVM model
12. Make predictions
13. Evaluate model performance
14. Experiment with different `C` values
15. Apply cross-validation
17. Tune hyperparameters using GridSearchCV
18. Build the final SVM model
19. Evaluate the final model

---

## ⚙️ Data Preprocessing

Unnecessary columns such as the record ID are removed before model training.

```python
df = df.drop("id", axis=1)
```

Features and target are separated using:

```python
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]
```

---

## ✂️ Train-Test Split

The dataset is divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The test set is kept separate for final model evaluation.

---

## 📏 Feature Scaling

Feature scaling is important for SVM because the features have different numerical ranges.

`StandardScaler` is used:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data.

---

## 🤖 Support Vector Machine

A baseline SVM model is created using a linear kernel:

```python
svm_linear = SVC(
    kernel="linear",
    C=1
)

svm_linear.fit(X_train_scaled, y_train)
```

SVM attempts to find a decision boundary that separates the classes while maximizing the margin between them.

---

## 🔧 Soft Margin and C Hyperparameter

The `C` hyperparameter controls the penalty applied to margin violations.

Different values of `C` are tested:

```python
c_values = [0.01, 0.1, 1, 10, 100]
```

Smaller values of `C` provide stronger regularization and allow more margin violations, while larger values penalize violations more strongly.

---

## 🔄 Cross-Validation

5-fold cross-validation is used to evaluate different hyperparameter settings more reliably.

```python
scores = cross_val_score(
    model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring="accuracy"
)
```

The mean cross-validation score is used to compare model configurations.



## 📈 Model Evaluation

The final model is evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score
- Classification Report

Example:

```python
print("Accuracy:", accuracy_score(y_test, y_pred_final))

print(
    classification_report(
        y_test,
        y_pred_final
    )
)
```

---

## 📊 Confusion Matrix

A confusion matrix is used to understand the types of classification errors made by the model.

```python
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred_final),
    display_labels=final_svm.classes_
).plot()

plt.show()
```

This helps identify how often benign and malignant cases are classified correctly or incorrectly.

---

## 💡 Key Learnings

Through this project, I practiced:

- Support Vector Machine classification
- Hyperplanes and margins
- Support vectors
- Hard-margin and soft-margin concepts
- Feature scaling
- SVM `C` hyperparameter
- Linear, RBF and Polynomial kernels
- Cross-validation
- Confusion matrix interpretation
- Precision, Recall and F1-score
- Model evaluation

---

## 🚀 Future Improvements

Possible future improvements include:

- Using an sklearn Pipeline for preprocessing and SVM
- Performing more extensive hyperparameter tuning
- Comparing SVM with KNN, Logistic Regression and tree-based models
- Adding feature selection
- Analyzing feature relationships in greater detail

---

## ⚠️ Disclaimer

This project is created for **machine learning education and portfolio purposes only**. It is not intended to provide medical diagnosis or replace professional medical evaluation.

---

## 👤 Author

**Shahzaib Abid**

Machine Learning & AI Learner

---

⭐ If you found this project useful, feel free to star the repository.
