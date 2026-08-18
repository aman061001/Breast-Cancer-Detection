"""Build five breast-cancer diagnosis classifiers and save deployment artifacts.

Execute from the repository root:
    python model/train_diagnosis_models.py
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_CSV = PROJECT_DIR / "dataset" / "breast_cancer_wisconsin.csv"
TEST_CSV = PROJECT_DIR / "test_data.csv"
OUTPUT_DIR = PROJECT_DIR / "model"
SUMMARY_JSON = OUTPUT_DIR / "diagnostic_summary.json"

TARGET = "diagnosis"
DROP_COLUMNS = ["id", "Unnamed: 32"]
POSITIVE_CLASS = "M"
NEGATIVE_CLASS = "B"
SEED = 17
TEST_FRACTION = 0.20

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "gaussian_naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load, validate, and clean the uploaded Wisconsin dataset."""
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE_CSV}")

    raw = pd.read_csv(SOURCE_CSV)
    required = {TARGET, "id"}
    missing_required = sorted(required - set(raw.columns))
    if missing_required:
        raise ValueError(f"Dataset is missing required columns: {missing_required}")

    clean = raw.drop(columns=[column for column in DROP_COLUMNS if column in raw.columns]).copy()
    clean = clean.drop_duplicates()

    if set(clean[TARGET].dropna().unique()) != {NEGATIVE_CLASS, POSITIVE_CLASS}:
        raise ValueError("diagnosis must contain exactly B and M labels")

    y = clean.pop(TARGET).map({NEGATIVE_CLASS: 0, POSITIVE_CLASS: 1}).astype(int)
    X = clean.apply(pd.to_numeric, errors="coerce")
    return X, y


def model_set() -> dict[str, object]:
    """Return independent pipelines with preprocessing fitted only on training data."""
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=SEED,
            ),
        ),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(
                max_depth=4,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
        "kNN": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7, weights="distance"),
        ),
        "Naive Bayes": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            GaussianNB(var_smoothing=1e-9),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=350,
                max_features="sqrt",
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
    }


def evaluate_binary(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
    malignant_probability: np.ndarray,
) -> dict[str, float]:
    """Calculate assignment metrics using malignant diagnosis as the positive class."""
    return {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "AUC": float(roc_auc_score(actual, malignant_probability)),
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "Recall": float(recall_score(actual, predicted, zero_division=0)),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
        "MCC": float(matthews_corrcoef(actual, predicted)),
    }


def build_artifacts() -> dict:
    """Create the split, fit every model, save test data, and write model metadata."""
    X, y = load_dataset()
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_FRACTION,
        random_state=SEED,
        stratify=y,
    )

    test_export = X_test.copy()
    test_export[TARGET] = y_test.map({0: NEGATIVE_CLASS, 1: POSITIVE_CLASS}).to_numpy()
    test_export.to_csv(TEST_CSV, index=False)

    results: list[dict[str, float | str]] = []
    for name, pipeline in model_set().items():
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        probability = pipeline.predict_proba(X_test)[:, 1]
        results.append({"Model": name, **evaluate_binary(y_test, prediction, probability)})

        model_path = OUTPUT_DIR / MODEL_FILENAMES[name]
        joblib.dump(pipeline, model_path, compress=3)
        restored = joblib.load(model_path)
        if not np.array_equal(prediction, restored.predict(X_test)):
            raise RuntimeError(f"Saved-model verification failed for {name}")

    result_frame = pd.DataFrame(results)
    summary = {
        "student": {
            "name": "Sangam Samdarshi",
            "bits_id": "2025AC05127",
        },
        "project": "Breast Cancer Wisconsin Diagnostic Classification",
        "dataset_url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "dataset": {
            "rows": int(len(X)),
            "features": feature_names,
            "feature_count": len(feature_names),
            "target": TARGET,
            "label_meaning": {
                "B": "Benign",
                "M": "Malignant",
            },
            "dropped_columns": DROP_COLUMNS,
        },
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_fraction": TEST_FRACTION,
            "random_state": SEED,
            "stratified": True,
        },
        "positive_class": {
            "encoded_value": 1,
            "original_label": POSITIVE_CLASS,
            "display_label": "Malignant",
        },
        "models": {
            "filenames": MODEL_FILENAMES,
            "results": results,
        },
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBreast cancer models trained successfully.\n")
    print(result_frame.round(4).to_string(index=False))
    print(f"\nSaved labelled test data: {TEST_CSV.relative_to(PROJECT_DIR)}")
    print(f"Saved model files directly under: {OUTPUT_DIR.relative_to(PROJECT_DIR)}")
    return summary


if __name__ == "__main__":
    build_artifacts()
