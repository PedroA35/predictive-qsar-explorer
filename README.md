# Predictive Therapeutic Target Explorer & QSAR Platform

This repository features a scalable, Python-driven Cheminformatics web platform designed to perform virtual screening of small molecular compounds. By leveraging Machine Learning models, the application predicts the biological activity ($IC_{50}$) of input chemical structures against specific therapeutic targets.

The architecture is built to be completely modular, allowing the dynamic loading of models for multiple enzymes. The initial pipeline has been successfully validated and integrated using **Thrombin (ChEMBL ID: CHEMBL204)** as the primary target baseline.

---

## 🚀 Key Features

* **Dual-Input Mode:** Look up chemical structures by compound name/synonym via the PubChem API or input SMILES strings directly.
* **Real-time SMILES Validation:** Automated parsing, verification, and sanitization of chemical structures using the RDKit library.
* **Molecular Vectorization:** Transformation of SMILES strings into numeric representations via *Morgan Fingerprints* (Radius 2, 2048 bits).
* **Intelligent Bioactivity Prediction:** An intuitive graphical user interface (GUI) that consumes pre-trained serialization models to classify compounds as Active or Inactive.
* **Scalable Architecture:** Built-in support to dynamically switch between different enzyme models through a central target selection dropdown menu.

---

## 📊 Methodology & Data Pipeline

The project follows a rigorous data science workflow split into two main building blocks:

### 1. Data Engineering & Model Training (`notebooks/`)
* **Data Retrieval:** Automated mining of bioactivity ($IC_{50}$) records using the official ChEMBL API client.
* **Data Curation & Curation:** Comprehensive noise filtering, handling of duplicate compounds, removal of invalid SMILES, and unit standardisation (normalization to nM). The final curated baseline dataset yielded **2,260 unique molecules** for Thrombin.
* **Activity Labelling:** Binary classification based on the standard literature threshold. Compounds with an $IC_{50} < 10\,\mu\text{M}$ are classified as **Active** (1), while the rest are labelled **Inactive** (0).
* **Model Exploration:** Implementation and optimization of decision-tree-based algorithms specifically tuned to overcome the native class imbalance of biological datasets. Three approaches were trained and compared:
  * Baseline Gradient Boosting
  * Tuned Random Forest (with balanced class weights)
  * XGBoost (optimized using `scale_pos_weight`)

### 2. Streamlit Web Deployment (`app.py`)
* The frontend web application is developed entirely in Python using **Streamlit**, avoiding heavy web frameworks or external JavaScript/HTML dependencies.
* Asynchronous consumption of the optimized classifier loaded via serialized `joblib` files (`.pkl`), turning raw structural data into immediate binary predictions.

---

## 📈 Model Performance (Thrombin Baseline)

The models were evaluated on an independent, stratified test set. The top-performing configuration was saved as `model_thrombin.pkl`.

* **Selected Architecture:** [Insert the winning model name here, e.g., XGBoost / Random Forest]
* **G-Mean Score:** [Insert your value here from the notebook, e.g., 0.84]
* **F1-Score (Active Class):** [Insert your value here from the notebook, e.g., 0.76]
* **PR-AUC:** [Insert your value here from the notebook, e.g., 0.81]

> *Note: Detailed confusion matrices, ROC curves, and Precision-Recall evaluations can be directly reviewed and reproduced inside the training notebook.*

---

## 🛠️ Local Installation & Usage

### Prerequisites
Make sure you have Python installed on your local machine. Using a virtual environment (such as `venv` or `conda`) is highly recommended.

### 1. Clone the repository
```bash
git clone [https://github.com/PedroA35/predictive-qsar-explorer.git](https://github.com/PedroA35/predictive-qsar-explorer.git)
cd predictive-qsar-explorer
