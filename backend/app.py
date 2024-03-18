from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

import csv
import PyPDF2
import docx
import os


#Model
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', return_dict=False)

#Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', return_dict=False)

def get_answer(question, file):

    try:
        
        # get file data
        file_data = get_file(file.filename)
        encoding = tokenizer.encode_plus(text=question,text_pair=file_data)

        inputs = encoding['input_ids']  #Token embeddings
        sentence_embedding = encoding['token_type_ids']  #Segment embeddings
        tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens

        start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
        start_index = torch.argmax(start_scores)

        end_index = torch.argmax(end_scores)

        response = ' '.join(tokens[start_index:end_index+1])

        #FORMMATING THE RESPONSE
        if "#" in response:
            
            # Replace '##' with an empty string
            formatted_response = response.replace('##', '')

            # Remove extra spaces
            formatted_response = formatted_response.replace(' ', '')
        else:
            formatted_response = response
        return formatted_response

    except Exception as e:
        print("Error in get_answer function: ", e)
        if isinstance(e, ValueError):
            return "Issue occurred while loading the file, Please retry with a different file!!"
        return ""

# file reading functions
MAX_FILE_SIZE_MB = 100  # Maximum file size in megabytes

def read_csv(file_path):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        data = '\n'.join(','.join(row) for row in csv_reader)
    return data

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        data = '\n'.join(page.extractText() for page in pdf_reader.pages)
    return data

def read_docx(file_path):
    doc = docx.Document(file_path)
    paragraphs = []
    for paragraph in doc.paragraphs:
        paragraphs.append(paragraph.text)
    data = '\n'.join(paragraphs)
    return data

def read_txt(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

def read_file(file_path):
    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to megabytes
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File size exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB")
    
    if file_path.endswith('.csv'):
        return read_csv(file_path)
    elif file_path.endswith('.pdf'):
        return read_pdf(file_path)
    elif file_path.endswith('.docx'):
        return read_docx(file_path)
    elif file_path.endswith('.txt'):
        return read_txt(file_path)
    else:
        raise ValueError("Unsupported file format")

def get_file(file):
    file_path =  file # Path to your file
    file_data = read_file(file_path)

    return file_data
class Response(BaseModel):
    result: str | None

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=Response)
async def predict(file: UploadFile = File(...), question: str = None) -> Any:

    response = get_answer(question, file)
    if response == "":
        response = "Please retry with a different question or file!!"
    
    return {"result": response}
