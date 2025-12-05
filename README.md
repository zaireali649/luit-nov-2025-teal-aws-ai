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
See `caip_03_01/caip_week3_call1.ipynb` for conceptual overview and learning objectives.

## Week 03 - Call 02: AWS Bedrock API Integration and Error Handling

### Topics Covered
- Bedrock Runtime client vs Bedrock client: when to use each
- Constructing request bodies for Claude models on Bedrock
- Request body format: `anthropic_version`, `max_tokens`, `messages` structure
- Invoking Bedrock models programmatically using `invoke_model()`
- Response parsing: extracting text from Bedrock API responses
- ThrottlingException: understanding rate limits and why they occur
- Exponential backoff retry logic for handling throttling
- Rate limiting strategies: adding delays between API calls
- Prompt templates for batch processing multiple requests
- Error handling best practices for production API integrations

### Hands-On Work
- Created Bedrock runtime client using `boto3.client('bedrock-runtime')`
- Built `construct_body()` function to format requests for Claude models
- Implemented `call_bedrock()` function for model invocation and response parsing
- Experienced ThrottlingException when processing 50 US states sequentially
- Implemented exponential backoff retry logic (`call_bedrock_with_retry()`)
- Added rate limiting with `time.sleep()` between requests
- Used prompt templates with placeholders for batch processing
- Processed all 50 US states to collect state flower data
- Parsed JSON responses from Bedrock API
- Saved structured data to JSON file

### Key Learnings
- **Critical distinction**: Use `bedrock-runtime` client for inference, not `bedrock` client
- **Request format**: Body must be JSON string (`json.dumps()`), include `anthropic_version: "bedrock-2023-05-31"`
- **Response parsing**: `response["body"].read()` returns bytes, must parse with `json.loads()`
- **Throttling is normal**: Even legitimate usage can trigger rate limits - always implement retry logic
- **Exponential backoff**: Formula `(2 ** attempt) + random.uniform(0, 1)` provides effective retry strategy
- **Rate limiting**: Always add delays (`time.sleep(1)`) between requests in loops
- **Production readiness**: Retry logic and rate limiting are essential for production code

### Notebook
See `caip_03_02/caip_week3_call2.ipynb` for hands-on implementation and error handling examples.

### Files
- **`caip_03_02/call_bedrock.py`**: Core functions for constructing request bodies and calling Bedrock models
- **`caip_03_02/run_bedrock.py`**: Batch processing script with retry logic and rate limiting for processing 50 US states
- **`caip_03_02/prompt_template.txt`**: Template with placeholders for structured JSON output requests
- **`caip_03_02/throttling_explanation.md`**: Detailed explanation of ThrottlingException encountered during batch processing
- **`caip_03_02/flower_data.json`**: Output file containing state flower data collected from Bedrock API