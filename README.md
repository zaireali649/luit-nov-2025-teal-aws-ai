# luit-nov-2025-teal-aws-ai

AWS AI Foundational Exam Preparation

## Week 01 - Call 01: Intro to AI/ML

### Topics Covered
- AI vs ML vs DL vs GenAI concepts and relationships
- Supervised vs unsupervised learning (with examples: loan approval, transaction clustering)
- Classification vs regression
- Underfitting vs overfitting (visualized with Decision Trees)
- Class imbalance in binary classification
- ML lifecycle: prepare, train, predict/evaluate

### Hands-On Work
- Built complete ML pipeline using Iris dataset
- Data preparation: train/test split, feature scaling
- Model training: DecisionTreeClassifier
- Evaluation: accuracy, classification report, confusion matrices
- Feature importance analysis
- Experiment: impact of removing most important feature

### Notebook
See `caip_01_01/caip_week1_call1.ipynb` for code examples and visualizations.

## Week 01 - Call 02: How Good Is Your Model?

### Topics Covered
- Model evaluation metrics: accuracy, precision, recall, F1 score
- Confusion matrix interpretation (raw and normalized)
- Why accuracy alone can be misleading, especially with imbalanced datasets
- Real-world decision-making: false positives vs false negatives
- Business context example: gun detection on Facebook Marketplace

### Hands-On Work
- Worked with Titanic dataset (Kaggle) - binary classification (survived/did not survive)
- Data cleaning: handling missing values, encoding categorical features
- Trained Decision Tree and Random Forest models
- Computed and compared metrics: accuracy, precision, recall, F1 score
- Visualized confusion matrices (raw counts and normalized)
- Interpreted classification reports and model performance

### Notebook
See `caip_01_02/caip_week1_call2.ipynb` for code examples and visualizations.

### Python Scripts
- **`caip_01_02/xgboost_model.py`**: XGBoost model implementation with 80/20 train/test split, confusion matrix, and classification report
- **`caip_01_02/optimized_model.py`**: Hyperparameter tuning script using RandomizedSearchCV for both XGBoost and Random Forest models. Automatically selects the best performing model and provides detailed evaluation metrics including cross-validation scores and feature importance.