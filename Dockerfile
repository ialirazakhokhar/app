FROM python:3.12

COPY . /app

WORKDIR /app


# Install the required Python packages
RUN pip install -r requirements.txt


# # Expose the port that your app runs on (default for FastAPI)
EXPOSE $PORT

# # Command to run your application (main.py or app.py)
# CMD ["python", "main.py"]  # Adjust if needed to match your entry point
CMD gunicorn --workers=2 --bind 0.0.0.0:$PORT app:app