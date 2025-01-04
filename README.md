# Dual-Mode Correlation System for Parkinson's Patients

Welcome to the **Dual-Mode Correlation System repository**! This system leverages both speech analysis and motion monitoring through machine learning models and wearable devices to detect Parkinson's disease symptoms and assess their severity.

---

## Prerequisites

- Python 3.8 or higher
- `pip` package manager
- Dataset files (placed in the `dataset` folder)

---

## Installation and Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/wixk7/dual-mode-parkinsons.git
cd dual-mode-parkinsons
```

### Step 2: Set Up a Virtual Environment
```bash
python3 -m venv .venv
.venv\Scripts\activate     
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Files and Their Purpose

- **app.py**: Main Python file to run the application.
- **parkinsons.ino**: Code for hardware integration (e.g., Arduino).
- **train_model.ipynb**: Jupyter notebook for training machine learning models.

### Pre-trained Models
- **gradient_boosting_model.pkl**
- **random_forest_model.pkl**
- **svc_model.pkl**
- **xgboost_model.pkl**

### Utility Files
- **scaler.pkl**: Used for feature scaling.
- **selector.pkl**: Feature selection utility.

### Dataset Folder
- Contains audio files used for testing and evaluation.

---

## Running the Application

### Step 1: Activate the Virtual Environment
```bash
.venv\Scripts\activate   
```

### Step 2: Run the Application
```bash
streamlit run app.py
```

### Step 3: Input Data
Upload audio files or provide the necessary data inputs as prompted by the application.

---

## Dataset Structure
Place your dataset files in the `dataset` directory, ensuring the structure matches the expected input format. For example:

```
dataset/
├── HC_AH/  # Healthy control samples
└── PD_AH/  # Parkinson's diagnosed samples
```

---

## Training the Model
If you need to retrain the model:

1. Open `train_model.ipynb` in Jupyter Notebook.
2. Follow the steps to preprocess data and train the models.
3. Save the newly trained model files in the root directory.

---

## Notes
- Ensure the dataset files are correctly structured and placed inside the `dataset` folder.
- The `.venv` folder is excluded from tracking in the `.gitignore` file to avoid unnecessary files in the repository.
- Pre-trained models are already included in this repository. Retraining is optional unless you modify the dataset or algorithms.

---

## Troubleshooting

### Common Issues

- **Missing Python Packages**: Ensure all dependencies are installed by running:
  ```bash
  pip install -r requirements.txt
  ```

- **Dataset Errors**: Check that the dataset is placed in the `dataset` folder and matches the expected structure.

---

## Contributions
Contributions to improve this application are welcome! Please create a pull request with your changes, and ensure proper documentation for any new features.

---



