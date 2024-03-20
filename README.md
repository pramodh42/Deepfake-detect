# Deepfake-detect
 # FastAPI Image Prediction API

This project demonstrates the implementation of a FastAPI-based API for image classification using a pre-trained deep learning model.

## Features

- Upload an image and receive predictions on its content
- Supports various image formats (e.g., JPEG, PNG)
- Utilizes a pre-trained deep learning model for image classification
- Provides automatic API documentation via Swagger UI
- Implements error handling and exception logging for robustness

## Requirements

- Python 3.x
- FastAPI
- PyTorch (for deep learning model)
- Pillow (PIL) for image processing
- Uvicorn or other ASGI server for deployment

## Installation

1. Clone the repository

2. Create a virtual environement
   ```bash
    python -m venv venv
    venv\scripts\activate
    ```

3. Install requirements.txt

   ```bash
    pip install -r requirements.txt
    ```

4. Update setting.py file

5. Configure aws credentials in cli

6. Make sure that you have created a table named table_name in dynamodb

7. Start the server
    ```bash
    uvicorn app:app --reload
    ```

8. Navigate to localhost:8000/docs to see the swagger ui.

