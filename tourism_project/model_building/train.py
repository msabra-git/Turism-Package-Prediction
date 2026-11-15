# for data manipulation
import pandas as pd
# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
# for model serialization and experiment tracking
import joblib
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")   # complete the code to set the MLflow tracking URI
mlflow.set_experiment("MLOps_experiment3")     # complete the code to set the MLflow experiment name (same as the dev experimentation cell)

# Ensure a local MLflow server is running on port 5000
try:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.search_experiments()
except Exception:
    print("Starting a local MLflow server process...")
    subprocess.Popen(["mlflow", "ui", "--port", "5000"])
    time.sleep(5)
    mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("MLOps_experiment3")

df = pd.read_csv("tourism_project/data/tourism.csv")
print("Dataset loaded successfully.")

# Clean extra indexing columns
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation"
]

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Optimized parameters to improve F1-score and Recall generalization
param_grid = {
    'xgbclassifier__n_estimators': [100, 150],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.03, 0.05, 0.1],
    'xgbclassifier__subsample': [0.7, 0.9],
    'xgbclassifier__colsample_bytree': [0.7, 0.9],
    'xgbclassifier__reg_lambda': [1.0, 5.0]
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    # Optimize search specifically scoring for F1
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log parameters and mean test scores
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_f1", mean_score)
            mlflow.log_metric("std_test_f1", std_score)

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Use a optimized threshold for robust F1 and Recall balancing
    classification_threshold = 0.40

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"]
    })


    # Save the model next to app.py so the Streamlit app can load it directly,
    # and log it as an MLflow artifact for traceability
    model_path = "tourism_project/deployment/best_tourism_package_model_v1.joblib"   # Specify the local file path (inside tourism_project/deployment/) where the trained model should be saved.
    joblib.dump(best_model, model_path)  # complete the code to save the model
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
