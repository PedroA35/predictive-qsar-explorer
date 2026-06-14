# 🧬 Predictive Therapeutic Target & QSAR Explorer

A web-based cheminformatics application that predicts the biological activity of chemical molecules against multiple therapeutic targets using machine learning (QSAR modelling).

Built with **Streamlit**, **RDKit**, and **scikit-learn / XGBoost**.

---

## Features

- **Molecule search** by common name (via PubChem API) or SMILES string
- **2D structure visualisation** with an interactive legend
- **Single-target prediction** — detailed activity prediction with confidence score
- **Multi-target comparison** — side-by-side prediction table across all available targets
- **Four therapeutic targets** supported out of the box:

| Target | ChEMBL ID | Disease Area | Model |
|--------|-----------|--------------|-------|
| Thrombin | CHEMBL204 | Cardiovascular / Thrombosis | RF |
| BACE-1 | CHEMBL4822 | Alzheimer's Disease | GB |
| CDK2 | CHEMBL301 | Cancer (Cell Cycle) | RF |
| EGFR | CHEMBL203 | Cancer (Growth Factor) | RF |

> RF = Random Forest · GB = Gradient Boosting

---

## How It Works

1. The user enters a molecule name or SMILES string
2. If a name is provided, the app queries **PubChem** to retrieve the canonical SMILES
3. **RDKit** generates Morgan Fingerprints (radius=2, 2048 bits) from the SMILES
4. Pre-trained ML models predict activity (Active / Inactive) for each target
5. Results are displayed with predicted probability and confidence bar

---

## Project Structure

```
qsar-explorer/
│
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
│
├── models/
│   ├── model_thrombin.pkl    # Trained model — Thrombin
│   ├── model_bace1.pkl       # Trained model — BACE-1
│   ├── model_cdk2.pkl        # Trained model — CDK2
│   └── model_egfr.pkl        # Trained model — EGFR
│
└── notebooks/
    └── qsar_training.ipynb   # Full training pipeline for all targets
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/qsar-explorer.git
cd qsar-explorer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

> **Note:** The pre-trained `.pkl` model files must be present in the `models/` folder. See the section below if you want to retrain them.

---

## Retraining the Models

All models can be retrained from scratch using the notebook `notebooks/qsar_training.ipynb`.

1. Open the notebook in Jupyter
2. Set `ACTIVE_TARGET` to the desired target (`"thrombin"`, `"bace1"`, `"cdk2"`, or `"egfr"`)
3. Run all cells — the best model is saved automatically to `models/`
4. Repeat for each target

The pipeline downloads bioactivity data (IC₅₀) directly from **ChEMBL**, cleans the dataset, generates Morgan Fingerprints, and compares three models (Gradient Boosting, Tuned Random Forest, XGBoost) before saving the best one.

---

## Methodology

### Data
- Bioactivity data (IC₅₀) downloaded from [ChEMBL](https://www.ebi.ac.uk/chembl/)
- Molecules labelled as **Active** (IC₅₀ < 10 µM) or **Inactive** (IC₅₀ ≥ 10 µM)
- Duplicates removed by taking the median IC₅₀ per molecule

### Features
- **Morgan Fingerprints** (radius=2, 2048 bits) generated with RDKit
- Captures circular substructure patterns around each atom

### Models
- **Baseline:** Gradient Boosting Classifier
- **Tuned RF:** Random Forest with `class_weight="balanced"` + RandomizedSearchCV
- **XGBoost:** with `scale_pos_weight` to handle class imbalance + RandomizedSearchCV
- Best model selected automatically by ROC-AUC on the test set

---

## Interpretation

- **Active** → predicted IC₅₀ < 10 µM — the molecule is likely to inhibit the target at pharmacologically relevant concentrations
- **Inactive** → predicted IC₅₀ ≥ 10 µM — low predicted potency for this target
- **P(active) %** → model confidence in the Active prediction
- Results are **computational predictions** and should always be validated experimentally

---

## Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Main libraries:

- [Streamlit](https://streamlit.io/) — web interface
- [RDKit](https://www.rdkit.org/) — cheminformatics and fingerprint generation
- [scikit-learn](https://scikit-learn.org/) — machine learning models
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosting
- [ChEMBL webresource client](https://github.com/chembl/chembl_webresource_client) — bioactivity data
- [PubChem API](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) — molecule name resolution

---

## Author

Developed by **Pedro** as part of a Bioinformatics project at FCUP.
