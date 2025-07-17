# Term Deposit Subscription Prediction

A machine learning project that predicts whether a client will subscribe to a term deposit based on personal and campaign-related attributes collected by a Portuguese banking institution.

## 📊 Project Overview

This project implements various machine learning algorithms to solve a binary classification problem: predicting whether a bank client will subscribe to a term deposit (yes/no) based on their demographic and campaign interaction data.

### Key Features

- **Multiple ML Models**: Logistic Regression, Random Forest, SVM, XGBoost, and LightGBM
- **Comprehensive EDA**: Exploratory data analysis with visualizations
- **Feature Engineering**: Advanced feature creation and selection
- **Hyperparameter Tuning**: Grid search optimization for best performance
- **Class Imbalance Handling**: SMOTE resampling technique
- **Web Application**: Interactive Streamlit app for predictions
- **Model Deployment**: Production-ready model saving and loading

## 🎯 Objective

The goal is to develop a machine learning model that can accurately predict term deposit subscriptions, helping banks optimize their marketing campaigns and improve customer targeting strategies.

## 📈 Performance

The best performing model (LightGBM) achieved:
- **Accuracy**: 90.05%
- **F1-Score**: Optimized for balanced performance
- **ROC AUC**: High discriminative ability

## 🗂️ Project Structure

```
TermDeposit/
├── .gitignore                # Ignore files for Git
├── LICENSE                   # Project license (MIT)
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/                     # Place the dataset here (not included)
│   └── README.md
├── models/                   # For saved models (not included)
│   └── README.md
├── notebooks/                # Jupyter notebooks
│   └── TermDepositPrediction.ipynb
```

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/beyzaasan/term-deposit-prediction
   cd term-deposit-prediction/TermDeposit
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the dataset:**
   - Download `bank-additional.csv` from the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
   - Place it in the `data/` directory

4. **Run the analysis**
   ```bash
   # Run the Jupyter notebook
   jupyter notebook notebooks/TermDepositPrediction.ipynb
   
   # Or run the Streamlit app
   streamlit run app.py
   ```

## Notes
- **Do not upload the dataset or model files to GitHub.**
- All code and analysis are in the notebook.
- For more details on data, models, and results, see the `README.md` files in each subdirectory.

The project uses the **Bank Marketing Dataset** from the UCI Machine Learning Repository:

- **Source**: [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Records**: 4,119
- **Features**: 20 input features + 1 target variable
- **Target**: Binary classification (`yes`/`no`)

### Feature Categories

| Type | Features |
|------|----------|
| **Demographic** | age, job, marital, education |
| **Financial** | default, housing, loan |
| **Campaign** | contact, month, day_of_week, duration, campaign |
| **Previous** | pdays, previous, poutcome |
| **Economic** | emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed |

## 🔧 Methodology

### 1. Data Preprocessing
- Handling missing values and inconsistencies
- Feature engineering (derived features)
- Encoding categorical variables
- Log transformations for skewed features

### 2. Exploratory Data Analysis
- Correlation analysis
- Mutual information scores
- Feature importance analysis
- Distribution visualizations
- Class imbalance analysis

### 3. Model Development
- Multiple algorithm comparison
- SMOTE resampling for class imbalance
- Hyperparameter tuning with GridSearchCV
- Cross-validation with stratified sampling

### 4. Model Evaluation
- Accuracy, Precision, Recall, F1-score
- ROC AUC analysis
- Classification reports
- Model comparison visualizations

## 🎯 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| LightGBM | 90.05% | 0.89 | 0.91 | 0.90 | 0.95 |
| XGBoost | 89.12% | 0.88 | 0.90 | 0.89 | 0.94 |
| Random Forest | 88.23% | 0.87 | 0.89 | 0.88 | 0.93 |
| SVM (RBF) | 82.52% | 0.81 | 0.84 | 0.82 | 0.89 |
| Logistic Regression | 76.70% | 0.75 | 0.78 | 0.76 | 0.84 |

### Key Insights

1. **LightGBM** performed best due to its ability to handle class imbalance and capture complex patterns
2. **Duration** and **previous campaign interactions** are the most important features
3. **Class imbalance** significantly affects model performance
4. **Feature engineering** improved model accuracy by 5-8%

## 🌐 Web Application

The project includes a Streamlit web application (`app.py`) that allows users to:

- Input customer information
- Get real-time predictions
- View prediction probabilities
- Understand feature importance

To run the app:
```bash
streamlit run app.py
```

## 📝 Usage Examples

### Training a New Model

```python
from src.model_training import train_models
from src.evaluation import evaluate_models

# Train all models
results = train_models(X_train, y_train, X_test, y_test)

# Evaluate and compare
evaluate_models(results)
```

### Making Predictions

```python
import joblib

# Load the trained model
model = joblib.load('models/best_model.pkl')

# Make predictions
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) for the dataset
- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [Streamlit](https://streamlit.io/) for the web application framework
- [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data manipulation

⭐ If you find this project helpful, please give it a star! 
