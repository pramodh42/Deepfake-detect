import bcrypt
import jwt
from datetime import datetime, timedelta
from boto3.dynamodb.conditions import Key
from fastapi import HTTPException,Header
from typing import Optional 
from settings import * 
from models import UserAuth 
import boto3

dynamodb = boto3.resource('dynamodb')


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def authenticate_user(user_name: str, password: str) -> Optional[str]:
    table = dynamodb.Table(table_name)
    response = table.query(
        KeyConditionExpression=Key('user_name').eq(user_name)
    )
    if response['Items'] and verify_password(password, response['Items'][0]['password']):
        access_token_expires = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = jwt.encode({"sub": user_name, "exp": access_token_expires}, SECRET_KEY, algorithm=ALGORITHM)
        return access_token  
    

    return None


async def get_token(user_auth: UserAuth):
    token = authenticate_user(user_auth.user_name, user_auth.password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return token

async def verify_token(token: str = Header(...)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_name: str = payload.get("sub")
        if user_name is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")