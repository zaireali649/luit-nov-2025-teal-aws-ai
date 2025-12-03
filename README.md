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

## Week 02 - Call 01: Train and Evaluate in the Cloud

### Topics Covered
- Benefits of cloud-based ML training: scalability, reproducibility, separation of concerns
- Infrastructure-as-Code (IaC) with CloudFormation for SageMaker Studio setup
- S3 integration for data and model artifact storage
- SageMaker SKLearn Estimator for managed training jobs
- CloudWatch logs for monitoring and debugging training jobs
- IAM roles and S3 permissions for secure ML workflows
- Model evaluation in cloud environment (confusion matrix, classification report, feature importance)

### Hands-On Work
- Deployed SageMaker Studio domain using CloudFormation template
- Created IAM execution role and S3 bucket programmatically
- Cleaned and prepared Titanic dataset
- Created standalone training script (`train_model.py`) for cloud execution
- Uploaded training data to S3
- Launched managed training job using SKLearn Estimator
- Reviewed training results in CloudWatch logs
- Compared cloud training output with local model for validation
- Programmatically cleaned up AWS resources

### Notebook
See `caip_02_01/caip_week2_call1.ipynb` for code examples and CloudFormation setup.

### Files
- **`caip_02_01/train_model.py`**: Training script that runs in SageMaker container, trains DecisionTreeClassifier, and saves model artifact
- **`caip_02_01/cf_templates/`**: CloudFormation templates for infrastructure setup

## Week 02 - Call 02: Deploy and Test Your First ML Endpoint

### Topics Covered
- Inference types in SageMaker: real-time, batch, and async inference
- Real-time endpoint architecture and deployment
- Loading model artifacts from S3
- SKLearnModel deployment for scikit-learn models
- Endpoint invocation using boto3 `invoke_endpoint` API
- CloudWatch logs for endpoint monitoring and debugging
- Cost management: deleting endpoints to avoid unnecessary charges
- Inference script structure (`model_fn`, `predict_fn`)

### Hands-On Work
- Located and loaded trained model artifact from S3
- Created inference script (`inference.py`) with model loading and prediction functions
- Deployed scikit-learn model to real-time SageMaker endpoint
- Sent test payloads using boto3 and interpreted prediction results
- Monitored endpoint activity in CloudWatch Logs
- Identified and reviewed endpoint metrics and logs
- Cleaned up endpoints programmatically

### Notebook
See `caip_02_02/caip_week2_call2.ipynb` for endpoint deployment and testing examples.

### Files
- **`caip_02_02/train_model.py`**: Training script (includes scaler saving for inference)
- **`caip_02_02/inference.py`**: Inference script with `model_fn` and `predict_fn` for SageMaker endpoint
- **`caip_02_02/cf_templates/`**: CloudFormation templates for infrastructure setup

## Week 03 - Call 01: Introduction to AWS Bedrock

### Topics Covered
- AWS Bedrock overview: fully managed service for foundational models
- Available foundational models: Anthropic Claude, AWS Titan, AI21 Jurassic, Meta Llama, Cohere
- Cost management: token-based pricing, warnings about Provisioned Throughput
- Model parameters: temperature (creativity vs consistency), top K, top P
- Prompt engineering techniques and best practices
- RAG (Retrieval-Augmented Generation) vs Fine-tuning approaches
- Career advice and best practices for working with foundational models

### Key Concepts
- **Token vs Word**: ~1000 tokens ≈ 750 words
- **Temperature**: Controls randomness (low = factual, high = creative)
- **Cost Control**: Keep max tokens at 2000 or below, monitor usage, set billing alerts
- **Provisioned Throughput Warning**: Avoid for learning - charges by hour regardless of usage

### Notebook
See `caip_03_01/caip_week3_call1.ipynb` for conceptual overview and learning objectives. Hands-on implementation will be covered in `caip_03_02`.