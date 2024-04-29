from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
import os
from typing import List
from langchain_core.documents import Document


import os
from PyPDF2 import PdfReader
import logging

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            path = os.path.join(directory, filename)
            text = load_pdf_document(path)
            if text:
                documents.append({'filename': filename, 'content': text})
        elif filename.endswith('.txt') or filename.endswith('.md'):
            path = os.path.join(directory, filename)
            text = load_text_document(path)
            if text:
                documents.append({'filename': filename, 'content': text})
        else:
            logging.warning(f"Ignored file '{filename}' as it is not a supported format.")
    return documents

def load_text_document(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        logging.error(f"Error reading text document at {path}: {e}")
        return None

def load_pdf_document(path):
    try:
        reader = PdfReader(path)
        text = ' '.join([page.extract_text() for page in reader.pages if page.extract_text() is not None])
        return text
    except Exception as e:
        logging.error(f"Error reading PDF document at {path}: {e}")
        return None

