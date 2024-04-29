from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from models import check_if_model_is_available
from document_loader import load_documents
import argparse
import sys

from llm import getChatChain

import logging

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def load_documents_into_database(model_name: str, documents_path: str) -> Chroma:
    """
    Loads documents from the specified directory into the Chroma database
    after splitting the text into chunks.

    Returns:
        Chroma: The Chroma database with loaded documents.
    """

    import os
    from PyPDF2 import PdfReader
    import mimetypes

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

    def load_documents(directory):
        documents = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                mimetype, _ = mimetypes.guess_type(filepath)
                if mimetype == 'application/pdf':
                    text = load_pdf_document(filepath)
                elif mimetype and mimetype.startswith('text'):
                    text = load_text_document(filepath)
                else:
                    logging.warning(f"Ignored file '{filename}' as it is not a supported format.")
                    continue
                
                if text:
                    documents.append({'filename': filename, 'content': text})
            else:
                logging.warning(f"Ignored non-file entry '{filename}'.")
        return documents

    print("Loading documents")
    raw_documents = load_documents(documents_path)
    if not raw_documents:
        raise ValueError("No documents loaded. Check the documents path and format.")

    documents = TEXT_SPLITTER.split_documents(raw_documents)
    if not documents:
        raise ValueError("Document splitting resulted in no output. Check the splitter and input documents.")

    print("Creating embeddings and loading documents into Chroma")
    try:
        db = Chroma.from_documents(
            documents,
            OllamaEmbeddings(model=model_name),
        )
    except Exception as e:
        print(f"Failed to load documents into Chroma: {e}")
        raise

    return db





def main(llm_model_name: str, embedding_model_name: str, documents_path: str) -> None:
    # Check to see if the models available, if not attempt to pull them
    try:
        check_if_model_is_available(llm_model_name)
        check_if_model_is_available(embedding_model_name)
    except Exception as e:
        print(e)
        sys.exit()

    # Creating database form documents
    try:
        db = load_documents_into_database(embedding_model_name, documents_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit()

    llm = Ollama(model=llm_model_name)
    chat = getChatChain(llm, db)

    while True:
        try:
            user_input = input(
                "\n\nPlease enter your question (or type 'exit' to end): "
            )
            if user_input.lower() == "exit":
                break

            chat(user_input)
        except KeyboardInterrupt:
            break


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local LLM with RAG with Ollama.")
    parser.add_argument(
        "-m",
        "--model",
        default="mistral",
        help="The name of the LLM model to use.",
    )
    parser.add_argument(
        "-e",
        "--embedding_model",
        default="nomic-embed-text",
        help="The name of the embedding model to use.",
    )
    parser.add_argument(
        "-p",
        "--path",
        default="Research",
        help="The path to the directory containing documents to load.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.model, args.embedding_model, args.path)
