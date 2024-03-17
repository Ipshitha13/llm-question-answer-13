**Question Answering System with FastAPI**


# Overview
- This application utilizes FastAPI to create a RESTful API for a Question Answering System. It provides a simple interface for users to upload files (in various formats such as PDF, DOCX, CSV, and TXT) containing text data, and ask questions related to the content of those files. The system then extracts the answer to the questions from the uploaded files using a pre-trained BERT model.
- I have used opensourced model from hugging-face, but you can try using openai GPT models OR any other opensource models with prompt instruction which takes the context of the file data and answer to user queries.


# Setup Instructions
- git clone <repository-url>
- Install Dependencies:
    - pip install -r requirements.txt
- Run the Application:
    - uvicorn main:app --reload
 
**Usage**
# Access API Documentation:
Once the server is running, you can access the API documentation at http://localhost:3000. This interactive documentation provides details on how to use each endpoint and what parameters are required.

# Upload Files:
Use the provided /predict endpoint to upload files and ask questions. You can upload files in formats such as PDF, DOCX, CSV, or TXT.

# Ask Questions:
Provide your question along with the uploaded file. The system will extract the answer to your question from the content of the file.

**Note**
- Ensure that the files you upload are within the maximum size limit of 100 MB.
- The application currently supports a limited set of file formats (PDF, DOCX, CSV, TXT).
