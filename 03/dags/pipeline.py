from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow.sklearn

DATA_PATH = '/opt/airflow/data/yellow_tripdata_2023-03.parquet'  # update if needed

def load_data():
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} records")
    df.to_parquet('/opt/airflow/data/df_loaded.parquet')

def preprocess_data():
    df = pd.read_parquet('/opt/airflow/data/df_loaded.parquet')
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)
    print(f"Filtered down to {len(df)} records")
    df.to_parquet('/opt/airflow/data/df_clean.parquet')

def train_model():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    df = pd.read_parquet('/opt/airflow/data/df_clean.parquet')
    dv = DictVectorizer()
    train_dicts = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration'].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"Model intercept: {model.intercept_:.2f}")

    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.sklearn.log_model(model, artifact_path="model")

dag = DAG(
    dag_id='taxi_pipeline_dag',
    start_date=datetime(2025, 6, 1),
    schedule=None,
    catchup=False
)

t1 = PythonOperator(task_id='load_data', python_callable=load_data, dag=dag)
t2 = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data, dag=dag)
t3 = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)

t1 >> t2 >> t3
