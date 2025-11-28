Use an official Python runtime as a parent image

FROM python:3.12-slim

Set the working directory in the container

WORKDIR /usr/src/app

Copy the current directory contents into the container at /usr/src/app

COPY . .

Install any needed packages specified in requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

Make port 8080 available to the world outside this container

EXPOSE 8080

Run gunicorn to serve the application, listening on 0.0.0.0:8080

The Flask app instance is named 'app' in app.py

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]