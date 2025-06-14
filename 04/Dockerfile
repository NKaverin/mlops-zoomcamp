FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

COPY Pipfile Pipfile.lock ./
RUN pip install -U pip
RUN pip install pipenv
RUN pip install numpy
RUN pipenv install --deploy --system

COPY starter.py .

CMD ["python", "starter.py"]
