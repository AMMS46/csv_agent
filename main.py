from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain_google_genai import GoogleGenerativeAI
import io
import os
from dotenv import load_dotenv


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


agent = None

class QueryRequest(BaseModel):
    query: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global agent
    try:
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=500, detail="Google API key not found")

        llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
        agent = create_csv_agent(llm, csv_buffer)
        return {"message": "File uploaded and agent created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_agent(request: QueryRequest):
    global agent
    if agent is None:
        raise HTTPException(status_code=400, detail="No agent available. Please upload a file first.")
    try:
        response = agent.run(request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
