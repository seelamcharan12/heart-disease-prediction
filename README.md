# ❤️ Heart Disease Prediction using Machine Learning

## 📌 About the Project

This project uses Machine Learning to predict whether a person is likely to have heart disease based on medical attributes available in the dataset.

The main goal is to understand how Machine Learning algorithms can learn patterns from data and make predictions.

**Note:** This project is developed for educational purposes and is not a substitute for professional medical advice or diagnosis.

## 🎯 Project Objectives

- Understand and preprocess a real-world-style heart disease dataset.
- Handle missing values in the dataset.
- Convert categorical data into a format suitable for Machine Learning.
- Train and evaluate different classification algorithms.
- Compare model performance and select a suitable model.
- Save the trained model for future predictions.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- VS Code

## 🤖 Machine Learning Algorithms

The project explores the following algorithms:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Hyperparameter tuning using GridSearchCV

## ⚙️ Project Workflow

1. Load the heart disease dataset.
2. Explore the dataset and identify missing values.
3. Separate input features and the target variable.
4. Split the data into training and testing sets.
5. Handle missing values using imputation.
6. Scale numerical features and encode categorical features.
7. Train Machine Learning models.
8. Evaluate the models using accuracy, confusion matrix, and classification report.
9. Tune model parameters using cross-validation.
10. Save the selected model using Joblib.

## 📊 Model Evaluation

The models are evaluated using:

- **Accuracy:** Measures the overall percentage of correct predictions.
- **Confusion Matrix:** Shows correct and incorrect predictions for each class.
- **Precision:** Measures how many predicted positive cases were actually positive.
- **Recall:** Measures how many actual positive cases the model identified.
- **F1-score:** Combines precision and recall into one metric.

Add the final accuracy values here after running your final version of `main.py`. Make sure the results match the code you upload.

## 📁 Project Structure

```text
heart-disease-prediction/
│
├── main.py
├── heart_disease_uci.csv
├── heart_disease_model.pkl
├── README.md
└── requirements.txt
```

The files shown above should be included only if they actually exist in your project folder.

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd heart-disease-prediction
```

### 3. Install the required libraries

```bash
pip install pandas numpy scikit-learn joblib
```

### 4. Run the Python program

```bash
python main.py
```

The program will train and evaluate the model and display the results implemented in the code.

## 🔮 Future Improvements

- Build a simple web interface for predictions.
- Improve model evaluation and validation.
- Add clearer explanations of prediction results.
- Test the model on appropriate external data.

## 👨‍💻 Conclusion

This project helped me gain practical experience with data preprocessing, classification algorithms, model evaluation, cross-validation, and model saving using Python and Scikit-learn.

It is an important step in my Machine Learning learning journey.

---

**Disclaimer:** Predictions from this project are for educational purposes only and must not be used to make medical decisions.