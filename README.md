# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using XGBoost classifier with handling for imbalanced datasets.

## Project Overview

This capstone project builds a binary classification model to identify fraudulent credit card transactions. The dataset is highly imbalanced (fraud cases are rare), so the project explores two approaches:
- **Standard approach**: Using class weight balancing
- **SMOTE approach**: Oversampling minority class to balance training data

## Project Structure

```
├── src/
│   ├── preprocess.py       # Data preprocessing and EDA
│   └── main.py             # Model training and evaluation
├── data/
│   ├── data_sources.txt    # Links to dataset sources
│   └── preprocessed/       # Generated preprocessed data
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Prerequisites

- Python 3.7+
- pip (Python package manager)

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/egecakiroglu/Credit_Card_Fraud_Detection.git
   cd Credit_Card_Fraud_Detection
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Getting the Data

The CSV data files are not included in this repository due to their large size. Download them from Kaggle using these datasets:

- https://www.kaggle.com/datasets/kartik2112/fraud-detection
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

**Steps to add data**:
1. Download the datasets from the links above
2. Extract and place the CSV files in the `data/` folder
3. For the main dataset, ensure you have `creditcard.csv`

## Running the Project

### Step 1: Preprocess the Data

```bash
python src/preprocess.py
```

This script will:
- Load the `creditcard.csv` file
- Analyze class distribution and data statistics
- Perform train-test split (70-30)
- Apply PCA for visualization
- Save preprocessed data to `data/preprocessed/creditcard_preprocessed.pkl`

**Output**:
- Data statistics and fraud percentage
- PCA visualization showing class separation
- Pickle file with train/test splits

### Step 2: Train and Evaluate Models

```bash
python src/main.py
```

This script will:
- Load preprocessed data
- Train XGBoost classifier without SMOTE
- Train XGBoost classifier with SMOTE oversampling
- Display evaluation metrics for both approaches
- Generate ROC and Precision-Recall curve visualizations

**Metrics displayed**:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC-AUC Score
- PR-AUC Score
- ROC and Precision-Recall curves

## Key Features

- **Imbalanced Data Handling**: Implements both `scale_pos_weight` and SMOTE techniques
- **Comprehensive Evaluation**: Uses multiple metrics beyond accuracy
- **Data Preprocessing**: Includes PCA visualization and proper train-test splitting
- **XGBoost Classifier**: Leverages gradient boosting for robust classification

## Dependencies

See `requirements.txt` for the full list. Key libraries:
- `pandas` & `numpy`: Data manipulation
- `scikit-learn`: Preprocessing and metrics
- `xgboost`: Gradient boosting classifier
- `imbalanced-learn`: SMOTE oversampling
- `matplotlib`: Visualization
- `joblib`: Model serialization

## Results

The project compares two approaches to handling class imbalance:

1. **Without SMOTE**: Balances class weights during training
2. **With SMOTE**: Generates synthetic samples to balance training data

Check the printed output and visualizations to compare ROC-AUC, PR-AUC, and other metrics between both approaches.

## Notes

- The dataset is highly imbalanced (~0.17% fraud cases), so standard accuracy is not a reliable metric
- Precision-Recall curves are often more informative for imbalanced classification
- ROC-AUC and PR-AUC are the primary evaluation metrics used

## Author

Capstone Project - 4th Year

## License

This project is open source. See LICENSE file if included.
