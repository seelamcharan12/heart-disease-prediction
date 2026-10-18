
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
data = pd.read_csv("heart_disease_uci.csv")

print("Dataset shape:", data.shape)
print(data.head())
print("Columns:", data.columns.tolist())

# 2. Create target
Y = (data["num"] > 0).astype(int)

# 3. Create input features
X = data.drop(columns=["id", "dataset", "num"], errors="ignore")

# 4. Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# 5. Identify numerical and categorical columns
num_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_columns = X_train.select_dtypes(exclude=["number"]).columns.tolist()

# 6. Preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_columns),
    ("cat", cat_pipeline, cat_columns)
])

# 7. Logistic Regression
lr_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

lr_model.fit(X_train, Y_train)
lr_pred = lr_model.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(Y_test, lr_pred) * 100)
print("Confusion Matrix:\n", confusion_matrix(Y_test, lr_pred))
print("Classification Report:\n",
      classification_report(Y_test, lr_pred, zero_division=0))

# 8. Decision Tree
dt_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
])

dt_model.fit(X_train, Y_train)
dt_pred = dt_model.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(Y_test, dt_pred) * 100)
print("Confusion Matrix:\n", confusion_matrix(Y_test, dt_pred))
print("Classification Report:\n",
      classification_report(Y_test, dt_pred, zero_division=0))

# 9. Random Forest with GridSearchCV
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf_grid = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=skf,
    scoring="accuracy",
    n_jobs=-1
)

rf_grid.fit(X_train, Y_train)

best_model = rf_grid.best_estimator_
rf_pred = best_model.predict(X_test)

print("\n--- Tuned Random Forest ---")
print("Best Parameters:", rf_grid.best_params_)
print("Best CV Accuracy:", rf_grid.best_score_ * 100)
print("Test Accuracy:", accuracy_score(Y_test, rf_pred) * 100)
print("Confusion Matrix:\n", confusion_matrix(Y_test, rf_pred))
print("Classification Report:\n",
      classification_report(Y_test, rf_pred, zero_division=0))

# 10. Save preprocessing + trained model together
joblib.dump(best_model, "heart_disease_model.pkl")
print("\nModel saved successfully!")

# 11. Load saved model
loaded_model = joblib.load("heart_disease_model.pkl")

# 12. Example prediction using a test row
new_data = X_test.iloc[[0]].copy()

prediction = loaded_model.predict(new_data)

print("\n--- Example Prediction ---")
print("Prediction:", int(prediction[0]))

if prediction[0] == 1:
    print("Dataset label: Disease")
else:
    print("Dataset label: No Disease")

# 13. Manual input prediction
# First check the exact column names printed above.
# Enter ALL feature columns, excluding id, dataset and num.
#
# Example structure:
#
# new_person = pd.DataFrame([{
#     "age": 50,
#     "sex": "Male",
#     "cp": "typical angina",
#     "trestbps": 130,
#     "chol": 240,
#     "fbs": False,
#     "restecg": "normal",
#     "thalch": 150,
#     "exang": False,
#     "oldpeak": 1.0,
#     "slope": "upsloping",
#     "ca": 0,
#     "thal": "normal"
# }])
#
# new_prediction = loaded_model.predict(new_person)
# print("New person's prediction:", int(new_prediction[0]))
