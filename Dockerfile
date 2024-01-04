FROM python:3.11-slim

WORKDIR /usr/src/app

# ENV NAME World
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY predict.py .
COPY pred_data.csv .

#copy model directory
COPY ./model/ ./model/

CMD ["python", "./predict.py"]