from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import zipfile
import os
import requests
import io
import json
import csv
import uvicorn
import pandas as pd
from PyPDF2 import PdfReader
import tempfile
from typing import Optional, List
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="File Processing API",
    description="API for processing ZIP files with OpenAI integration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "File Processing API is running. Send POST requests to / endpoint."}

@app.post("/")
async def process_file(file: UploadFile = File(None), question: str = Form(...)):
    logger.info(f"Processing request. Question: {question}, File: {file.filename if file else 'No file provided'}")
    
    if not file:
        return await process_question_only(question)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            content = await file.read()
            file_extension = os.path.splitext(file.filename)[1].lower()

            if file_extension == ".zip":
                return await process_zip_file(content, question, temp_dir)

            return await process_single_file(content, question, file.filename, temp_dir)
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"answer": f"Error: {str(e)}"}
            )

async def process_question_only(question: str) -> dict:
    try:
        logger.info("No file provided. Answering question directly.")
        
        systemPrompt = """
        You are a highly intelligent and precise assistant. Your role is to answer questions accurately and concisely based solely on the question text provided.

        If no additional content is given, use your general knowledge and reasoning skills to provide the exact answer without any explanations, context, or extra words.

        Always return ONLY the answer in its simplest and most direct form.
        """
        
        AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
        PROXY_URL = "https://aiproxy.sanand.workers.dev/openai"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }

        data = {
            "model" : "gpt-4o-mini",
            "messages" : [
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": question}
            ]
        }    

        response = requests.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=headers,
            json=data
        )
        response = response.json()
        answer = response["choices"][0]["message"]["content"].strip()
        return {"answer": answer}
    
    except Exception as e:
        logger.error(f"Error processing question-only request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

async def process_zip_file(content: bytes, question: str, temp_dir: str) -> dict:
    """
    Process a ZIP file and extract information using OpenAI.
    """
    zip_buffer = io.BytesIO(content)
    
    try:
        with zipfile.ZipFile(zip_buffer) as zip_file:
            zip_file.extractall(temp_dir)
            
            for file_name in zip_file.namelist():
                file_path = os.path.join(temp_dir, file_name)

                if os.path.isdir(file_path):
                    continue
                
                content_str = await read_file_content(file_path)
                answer = await query_openai(content_str, question)
                return {"answer": answer}
            
            return {"answer": "No valid files found in the ZIP"}
    
    except zipfile.BadZipFile:
        return {"answer": "Invalid ZIP file"}

async def process_single_file(content: bytes, question: str, filename: str, temp_dir: str) -> dict:
    try:
        temp_file_path = os.path.join(temp_dir, filename)
        with open(temp_file_path, "wb") as f:
            f.write(content)
        
        content_str = await read_file_content(temp_file_path)
        
        if not content_str.strip():
            return {"answer": "The uploaded file is empty or unsupported"}
        
        answer = await query_openai(content_str, question)
        return {"answer": answer}
    
    except Exception as e:
        logger.error(f"Error processing single file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing single file: {str(e)}")

async def read_file_content(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == ".pdf":
            with open(file_path, "rb") as f:
                return "".join(page.extract_text() for page in PdfReader(f).pages)
                
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path, dtype=str)
            return df.to_csv(index=False)
            
        elif ext == ".csv":
            df = pd.read_csv(file_path, dtype=str)
            return df.to_csv(index=False)
            
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                return json.dumps(json.load(f), indent=2)
                
        elif ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
                
        else:
            logger.warning(f"Unsupported file format: {ext}")
            return ""
    
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        return ""

async def query_openai(content: str, question: str) -> str:
    try:
        userPrompt = f"""
        Analyze the following content and answer this question: {question}

        Content:
        {content if content.strip() else "No content provided. Please answer based on the question alone."}

        Provide ONLY the exact answer value without any explanations or additional text.
        """
        systemPrompt = """
        You are a highly intelligent and precise assistant. Your role is to answer questions accurately and concisely based solely on the question text provided. 

        If no additional content is given, use your general knowledge and reasoning skills to provide the exact answer without any explanations, context, or extra words. 

        Always extract only the exact answer without any explanations and return ONLY the answer in its simplest and most direct form.
        """

        AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
        PROXY_URL = "https://aiproxy.sanand.workers.dev/openai"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }

        data = {
            "model" : "gpt-4o-mini",
            "messages" : [
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userPrompt}
            ]
        }    

        response = requests.post(
            f"{PROXY_URL}/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        response = response.json()
        answer = response["choices"][0]["message"]["content"].strip()
        return answer
    except Exception as e:
        logger.error(f"Error querying OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing with AI: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
