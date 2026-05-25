# 🏦 Credit Risk Prediction — FastAPI + Random Forest + AWS Lambda

A machine learning pipeline that predicts loan default risk using a Random Forest classifier, served via a **FastAPI REST API** with a web UI, Dockerized deployment, and AWS Lambda serverless support.

> **Live API Endpoint:** `POST /predict` — accepts borrower data in JSON and returns `Defaulted` / `Not Defaulted`

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Getting Started](#getting-started)
3. [Jupyter Notebook Analysis](#jupyter-notebook-analysis)
4. [FastAPI Prediction API](#fastapi-prediction-api)
5. [Web UI](#web-ui)
6. [Docker Deployment](#docker-deployment)
7. [AWS Lambda Deployment](#aws-lambda-deployment)

---

## Project Structure

```plaintext
Credit-Risk-Prediction/
├── app/
│   ├── app.py                          # FastAPI server — prediction endpoints
│   ├── Dockerfile                      # Docker setup
│   ├── requirements.txt                # API dependencies
│   ├── static/
│   │   └── style.css                   # Web UI stylesheet
│   ├── templates/
│   │   └── index.html                  # Web UI template
│   ├── ML_artifact/
│   │   └── RandomForest_Best.sav       # Serialized Random Forest model
│   └── data/
│       ├── Sample_input1.json          # Example API input
│       └── Sample_input2.json          # Example API input
├── Credit_Risk_Modelling.ipynb         # EDA, modeling, and evaluation notebook
├── Requirements.txt                    # Notebook dependencies
└── README.md
````

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/NirajanKhadka/Credit_Card_Risk_Modelling
cd Credit_Card_Risk_Modelling

# Install dependencies
pip install -r requirements.txt
```

---

## Jupyter Notebook Analysis

The notebook covers the full ML workflow — EDA, preprocessing, model training, and evaluation.

```bash
jupyter notebook Credit_Risk_Modelling.ipynb
```

**Notebook Highlights:**
- Exploratory data analysis and feature engineering
- Missing value handling and feature scaling
- Multiple classifiers compared (Random Forest selected as best)
- Evaluation: Accuracy, Precision, Recall, F1-score

---

## FastAPI Prediction API

### Input Schema

The API accepts an 11-feature JSON payload representing a loan applicant:

```json
{
  "person_age": 22,
  "person_income": 59000,
  "person_home_ownership": "RENT",
  "person_emp_length": 23.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "D",
  "loan_amnt": 35000,
  "loan_int_rate": 16.02,
  "loan_percent_income": 0.59,
  "cb_person_default_on_file": "Y",
  "cb_person_cred_hist_length": 3
}
```

### Run Locally

```bash
cd app
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Access the interactive API docs at `http://localhost:8000/docs`

### Sample Prediction Endpoint

```python
@app.post("/predict")
def predict(inference_request: Credit_Model):
    input_df = pd.DataFrame([inference_request.dict()])
    prediction = RF_pipeline.predict(input_df)
    result = "Not Defaulted" if prediction == 0 else "Defaulted"
    return {"Prediction": result}
```

---

## Web UI

After starting the server, navigate to `http://localhost:8000` to upload a JSON file and get predictions via the browser interface.

<p align="center"><img src="images/UI.png"></p>

---

## Docker Deployment

```bash
# Build the image
docker build -t credit_status:ml .

# Run the container
docker run -p 80:80 credit_status:ml
```

The API will be live at `http://localhost:80/predict`.

**Dockerfile summary:**
- Base: `python:3.8-slim`
- Exposes port 80
- Runs via `uvicorn main:app --host 0.0.0.0 --port 80`

<p align="center"><img src="images/DockerStartup.png"></p>

### Testing with Postman

Send a `POST` request to `http://localhost:80/predict` with raw JSON body.
Use **Newman** for CLI-based automation in CI/CD pipelines:

```bash
newman run your_collection.json
```

<p align="center"><img src="images/Postman.png"></p>

---

## AWS Lambda Deployment

Deploys the API as a serverless function using the **Serverless Framework**, with the model stored in **S3**.

### Prerequisites
- AWS CLI configured
- Serverless Framework installed (`npm install -g serverless`)
- S3 bucket created

### Steps

**1. Upload model to S3:**
```bash
aws s3 cp app/ML_artifact/RandomForest_Best.sav s3://loan-prediction-model-bucket/
```

**2. `serverless.yml` configuration:**
```yaml
service: loan-prediction-api

provider:
  name: aws
  runtime: python3.8
  region: us-east-1

functions:
  predict:
    handler: lambda_function.lambda_handler
    events:
      - http:
          path: predict
          method: post
          cors: true
    environment:
      MODEL_S3_BUCKET: loan-prediction-model-bucket
      MODEL_S3_KEY: RandomForest_Best.sav
```

**3. Deploy:**
```bash
serverless create --template aws-python3 --path my-service
serverless deploy
```

After deployment, API Gateway provides a live HTTPS endpoint you can hit directly via Postman or your frontend.

<p align="center"><img src="images/Serverless_deploy.png"></p>
