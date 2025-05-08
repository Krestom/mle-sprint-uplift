FROM python:3.9-slim

WORKDIR /app

# установим необходимые библиотеки
RUN apt-get update && apt-get install -y libgomp1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
COPY model.pkl .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]