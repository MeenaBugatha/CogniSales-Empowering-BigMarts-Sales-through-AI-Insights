FROM python:3.9.5-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
ENV FLASK_APP=big.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_SECRET_KEY="meenabuga"
CMD ["flask", "run", "--host=0.0.0.0"]
