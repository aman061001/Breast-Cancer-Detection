# Machine Learning Assignment 2 — Breast Cancer Wisconsin

**Student:** Sangam Samdarshi  
**BITS ID:** 2025AC05127  
**Dataset:** Breast Cancer Wisconsin Diagnostic  
**GitHub repository:** `https://github.com/sangamsamdarshi02/Breast-Cancer-Detection`  
**Live Streamlit app:** `REPLACE_WITH_SANGAM_STREAMLIT_APP_URL`

## A. Problem Statement

The objective is to classify a tumour record as benign or malignant using 30 numeric measurements calculated from digitized images of cell nuclei. The same fixed train/test split is used for all five classifiers explicitly listed in the assignment. Each model is evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

For metric calculation, malignant diagnosis is treated as the positive class. The Streamlit application provides CSV upload, model selection, saved model comparison, live evaluation, confusion matrix, classification report, and downloadable predictions.

> **Educational-use notice:** This project is an academic machine-learning demonstration. It is not a medical device and must not be used for diagnosis, treatment, or clinical decision-making.

## B. Dataset Description

**Public source:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The supplied CSV contains:

- 569 records
- 33 raw columns
- 30 usable numeric diagnostic features
- Target column: `diagnosis`
- Benign records: 357
- Malignant records: 212

Two non-feature columns are removed:

- `id`: record identifier
- `Unnamed: 32`: completely empty column

The target is encoded during training as:

| Original label | Encoded value | Meaning |
|---|---:|---|
| B | 0 | Benign |
| M | 1 | Malignant |

### Data preparation

1. Load the supplied CSV.
2. Remove `id` and `Unnamed: 32`.
3. Encode benign as 0 and malignant as 1.
4. Create one stratified 80/20 split using random state 17.
5. Train all models only on the 455 training records.
6. Save the untouched 114-row test partition as `test_data.csv` using the original B/M labels.
7. Keep median imputation inside every scikit-learn pipeline.
8. Apply standard scaling to Logistic Regression, kNN, and Gaussian Naive Bayes.
9. Leave Decision Tree and Random Forest unscaled.
10. Save every complete fitted pipeline as a separate Joblib file.

## C. GitHub Repository Link

`https://github.com/sangamsamdarshi02/Breast-Cancer-Detection`

## D. Models Used and Results

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9737 | 0.9974 | 0.9535 | 0.9762 | 0.9647 | 0.9439 |
| Decision Tree | 0.9386 | 0.9668 | 0.9070 | 0.9286 | 0.9176 | 0.8689 |
| kNN | 0.9737 | 0.9787 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| Naive Bayes | 0.9474 | 0.9927 | 0.9500 | 0.9048 | 0.9268 | 0.8864 |
| Random Forest (Ensemble) | 0.9561 | 0.9944 | 0.9512 | 0.9286 | 0.9398 | 0.9054 |

### Model observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall balance. It achieved the highest AUC, Recall, and F1 while maintaining Accuracy above 0.97 and MCC above 0.94. High recall is valuable in this educational comparison because it means fewer malignant records are missed. |
| Decision Tree | The simplest model to interpret but the weakest of the five on this split. Limiting depth and leaf size reduces overfitting, although a single tree still produces less stable boundaries than the ensemble and scaled linear model. |
| kNN | Achieved the same Accuracy as Logistic Regression, perfect Precision, and the highest MCC by a very small margin. However, its Recall and AUC were lower, meaning it missed more malignant records and ranked risk less consistently. |
| Naive Bayes | Fast and produced a very high AUC, but its conditional-independence assumption is not fully realistic because many tumour measurements are correlated. Its malignant Recall and F1 were below the leading models. |
| Random Forest (Ensemble) | Strong nonlinear classifier with high AUC and balanced scores. It outperformed the single Decision Tree but did not exceed Logistic Regression or kNN on the final test split. It also creates a larger and less transparent model. |
| Overall Winner | **Logistic Regression** because it provides the strongest combination of AUC, malignant Recall, F1, Accuracy, and MCC. |

## Streamlit Application

The interface is intentionally small and clinical in appearance. It uses a sidebar rather than the horizontal workflow used in the mobile-price project.

The app provides:

- Model selection dropdown
- Bundled `test_data.csv` or uploaded CSV
- Saved comparison table for all five models
- Six selected-model metrics
- Live metrics when the uploaded file contains `diagnosis`
- Benign/malignant confusion matrix
- Expandable classification report
- Prediction preview
- Downloadable prediction CSV
- Clear handling of unlabelled data and one-class uploads

The app accepts original `B`/`M` labels or numeric `0`/`1` labels. It validates all 30 required feature columns before prediction.

## Repository Structure

```text
sangam_breast_cancer_assignment/
├── app.py
├── README.md
├── requirements.txt
├── test_data.csv
├── dataset/
│   └── breast_cancer_wisconsin.csv
├── model/
│   ├── train_diagnosis_models.py
│   ├── diagnostic_summary.json
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── gaussian_naive_bayes.joblib
│   └── random_forest.joblib
├── analysis/
│   └── breast_cancer_experiment.ipynb
└── .streamlit/
    └── config.toml
```

## Run Locally on macOS

Open Terminal in this project folder and create a virtual environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Recreate the fixed split, metrics, JSON summary, and five model files:

```bash
python model/train_diagnosis_models.py
```

Run the notebook:

```bash
python -m pip install jupyterlab
jupyter lab
```

Open `analysis/breast_cancer_experiment.ipynb`, select the project environment, and run every cell.

Start the app:

```bash
streamlit run app.py
```

Open the local address displayed in Terminal, normally `http://localhost:8501`.

## Test the App Before Deployment

1. Confirm the comparison table shows five models and all six metrics.
2. Select each model from the sidebar.
3. Use the bundled `test_data.csv` and confirm live metrics reproduce the saved values.
4. Open the classification report.
5. Download the prediction CSV and open it.
6. Upload a CSV without `diagnosis` and confirm prediction-only mode works.
7. Remove one required feature and confirm the app reports the missing column.
8. Upload a one-class labelled subset and confirm AUC is shown as unavailable instead of crashing.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository using Sangam's GitHub account.
2. Commit and push this project to branch `main`.
3. Open Streamlit Community Cloud and create a new app.
4. Select the GitHub repository.
5. Select branch `main`.
6. Select `app.py` as the entrypoint.
7. Select Python 3.13 in advanced settings.
8. Deploy the app and inspect the build log.
9. Test the public link using bundled and uploaded data.
10. Replace the two URL placeholders at the top of this README and push the final update.

## Suggested Git Commands

```bash
git init -b main
git add .
git commit -m "Build breast cancer classification experiment"
git remote add origin REPLACE_WITH_SANGAM_GITHUB_REPOSITORY_URL
git push -u origin main
```

Make later commits after genuine work, such as completing model comparison, building the Streamlit evaluator, improving validation, and fixing deployment issues.

## BITS Virtual Lab and Submission PDF

Clone the same repository in the BITS Virtual Lab, install `requirements.txt`, run the training script or notebook, and capture one readable execution screenshot. The final PDF must contain, in this order:

1. GitHub repository link
2. Live Streamlit application link
3. One BITS Virtual Lab execution screenshot
4. Complete final README content

## Limitations

- Results depend on one fixed stratified split and the documented hyperparameters.
- The dataset is relatively small, so performance may vary on new populations.
- Correlated diagnostic features can affect models that make stronger independence assumptions.
- The application is strictly for educational demonstration and not clinical use.
