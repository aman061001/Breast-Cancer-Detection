"""Minimal Streamlit evaluator for the Breast Cancer Wisconsin classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE = Path(__file__).resolve().parent
MODEL_FOLDER = BASE / "model"
SUMMARY_PATH = MODEL_FOLDER / "diagnostic_summary.json"
SAMPLE_TEST_PATH = BASE / "test_data.csv"
TARGET = "diagnosis"

st.set_page_config(
    page_title="Diagnostic Model Review",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f7fafa; }
    .clinical-title {
        border-top: 6px solid #167d73;
        background: white;
        padding: 1rem 1.15rem;
        border-radius: 8px;
        box-shadow: 0 4px 14px rgba(25, 62, 68, .08);
        margin-bottom: 1rem;
    }
    .clinical-title h1 { margin: 0; color: #18323a; font-size: 2rem; }
    .clinical-title p { margin: .3rem 0 0 0; color: #47656a; }
    .disclaimer {
        border: 1px solid #c8dedd;
        background: #eef7f6;
        padding: .75rem .9rem;
        border-radius: 8px;
        color: #24494d;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #d9e7e6;
        border-radius: 8px;
        padding: .55rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_sample_test() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_TEST_PATH)


@st.cache_resource(show_spinner=False)
def read_model(filename: str):
    return joblib.load(MODEL_FOLDER / filename)


def as_score(value: float | None) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value:.4f}"


def normalise_input(
    frame: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    if frame.empty:
        raise ValueError("The selected CSV has no rows.")

    working = frame.copy()
    working.columns = [str(column).strip() for column in working.columns]

    raw_target = None
    if TARGET in working.columns:
        target_values = working[TARGET]
        if pd.api.types.is_numeric_dtype(target_values):
            numeric_target = pd.to_numeric(target_values, errors="coerce")
            if numeric_target.isna().any() or not set(numeric_target.astype(int).unique()).issubset({0, 1}):
                raise ValueError("Numeric diagnosis values must be 0 for benign or 1 for malignant.")
            raw_target = numeric_target.astype(int)
        else:
            cleaned_target = target_values.astype(str).str.strip().str.upper()
            invalid = sorted(set(cleaned_target.unique()) - {"B", "M"})
            if invalid:
                raise ValueError(f"Unexpected diagnosis labels: {invalid}")
            raw_target = cleaned_target.map({"B": 0, "M": 1}).astype(int)

    missing = [feature for feature in feature_names if feature not in working.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    numeric_features = working[feature_names].apply(pd.to_numeric, errors="coerce")
    return numeric_features, raw_target, working


def live_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
    malignant_probability: np.ndarray,
) -> dict[str, float | None]:
    auc_value: float | None
    try:
        auc_value = float(roc_auc_score(actual, malignant_probability))
    except ValueError:
        auc_value = None

    return {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "AUC": auc_value,
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "Recall": float(recall_score(actual, predicted, zero_division=0)),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
        "MCC": float(matthews_corrcoef(actual, predicted)),
    }


def metric_grid(values: dict[str, float | None]) -> None:
    for row_names in (("Accuracy", "AUC", "MCC"), ("Precision", "Recall", "F1")):
        row = st.columns(3)
        for column, name in zip(row, row_names):
            column.metric(name, as_score(values[name]))


def prediction_output(
    source: pd.DataFrame,
    prediction: np.ndarray,
    malignant_probability: np.ndarray,
) -> pd.DataFrame:
    output = pd.DataFrame(index=source.index)
    if "id" in source.columns:
        output["id"] = source["id"].to_numpy()
    output["predicted_diagnosis"] = np.where(prediction == 1, "M", "B")
    output["predicted_label"] = np.where(prediction == 1, "Malignant", "Benign")
    output["probability_benign"] = 1.0 - malignant_probability
    output["probability_malignant"] = malignant_probability
    if TARGET in source.columns:
        output["actual_diagnosis"] = source[TARGET].astype(str).to_numpy()
    return output


try:
    summary = read_summary()
except Exception as error:
    st.error(
        "Model artifacts are unavailable. Run `python model/train_diagnosis_models.py` "
        "from the repository root before starting Streamlit."
    )
    st.exception(error)
    st.stop()

model_files = summary["models"]["filenames"]
model_names = list(model_files)
feature_names = list(summary["dataset"]["features"])
comparison = pd.DataFrame(summary["models"]["results"])

st.markdown(
    f"""
    <div class="clinical-title">
        <h1>Diagnostic Model Review</h1>
        <p>Breast Cancer Wisconsin classification · {summary['student']['name']} · {summary['student']['bits_id']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer">
    <strong>Academic demonstration only.</strong> This application compares machine-learning
    classifiers on a public dataset. It is not a medical device and must not be used for
    clinical diagnosis or treatment decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Evaluation controls")
selected_name = st.sidebar.selectbox("Model", model_names)
data_source = st.sidebar.radio("CSV source", ["Bundled test data", "Upload CSV"])
uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
show_preview = st.sidebar.checkbox("Show input preview", value=False)

st.subheader("Saved test-set comparison")
display_table = comparison.copy()
for column in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
    display_table[column] = display_table[column].map(lambda value: f"{value:.4f}")
st.dataframe(display_table, hide_index=True, use_container_width=True)

saved_row = comparison.loc[comparison["Model"] == selected_name].iloc[0]
st.markdown(f"### Selected model: {selected_name}")
metric_grid({name: float(saved_row[name]) for name in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]})

if data_source == "Bundled test data":
    input_frame = read_sample_test()
else:
    if uploaded_file is None:
        st.info("Upload a compatible CSV from the sidebar to run predictions.")
        st.stop()
    input_frame = pd.read_csv(uploaded_file)

if show_preview:
    st.markdown("#### Input preview")
    st.dataframe(input_frame.head(25), use_container_width=True)

try:
    X_input, y_input, original_frame = normalise_input(input_frame, feature_names)
except ValueError as error:
    st.error(str(error))
    st.stop()

pipeline = read_model(model_files[selected_name])
predicted = pipeline.predict(X_input)
malignant_probability = pipeline.predict_proba(X_input)[:, 1]
output = prediction_output(original_frame, predicted, malignant_probability)

st.divider()
st.subheader("Results for the selected CSV")

if y_input is not None:
    scores = live_metrics(y_input, predicted, malignant_probability)
    metric_grid(scores)
    if scores["AUC"] is None:
        st.info("AUC is unavailable because this upload contains only one diagnosis class.")

    figure, axis = plt.subplots(figsize=(5.2, 4.0))
    ConfusionMatrixDisplay.from_predictions(
        y_input,
        predicted,
        labels=[0, 1],
        display_labels=["Benign", "Malignant"],
        cmap="Greens",
        colorbar=False,
        values_format="d",
        ax=axis,
    )
    axis.set_title("Confusion matrix")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=False)
    plt.close(figure)

    with st.expander("Open classification report"):
        report = classification_report(
            y_input,
            predicted,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
else:
    st.info(
        "No diagnosis column was found, so the app is showing predictions only. "
        "Evaluation metrics require known B/M labels."
    )

st.markdown("#### Prediction preview")
st.dataframe(output.head(50), hide_index=True, use_container_width=True)
st.download_button(
    "Download prediction CSV",
    data=output.to_csv(index=False).encode("utf-8"),
    file_name="breast_cancer_predictions.csv",
    mime="text/csv",
)

st.caption(
    "Malignant is encoded as the positive class for Precision, Recall, F1, and AUC. "
    "Uploaded records are never used to retrain the saved models."
)
