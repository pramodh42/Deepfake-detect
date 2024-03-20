from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header,Request 
from fastapi.responses import JSONResponse
from typing import Optional
from models import *
from service import * 
from settings import *
import boto3
import io
import torch
from torchvision.transforms import transforms
from PIL import Image
from Torch.model import Model,transform,device

app = FastAPI(title="Deep Fake Detection")

@app.post("/signup")
async def signup(user_create: UserCreate):
    hashed_password = hash_password(user_create.password)
    table = dynamodb.Table(table_name)

    table.put_item(Item=UserAuth(UserCreate))
    return {"message": "Registration successful"}

@app.post("/login", response_model=Token)
async def login(user_auth: UserAuth):
    token = await get_token(user_auth) 

    return {"access_token": token, "token_type": "bearer"}

# @app.post("/predict")
# async def predict(file: UploadFile = File(...), token: str = Depends(verify_token)):
#     return {"prediction": "Prediction result"}

def predict_image(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = Model.model(image_tensor)
        prob = outputs[0].item()
        return prob > 0.5

@app.post("/predict")
async def predict(file: UploadFile = File(...),Token: str = Header(...)):
    decoded_token = await verify_token(Token)
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    # return {"User agent":user_agent}
    return {"prediction": predict_image(img)} 

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": detail},
    ) 
@app.exception_handler(Exception)
async def http_exception_handler(request: Request, exc: Exception):
     error_details = str(exc) 
     return JSONResponse(
        content={"detail": error_details},
    ) 
