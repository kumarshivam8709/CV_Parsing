from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI

import os
import json
from dotenv import load_dotenv

load_dotenv('key.env')
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

def parse_result_string(result_str):
    extracted_data = {}
    for line in result_str.split('\n'):
        if ':' in line:
            key, value = line.split(': ', 1)
            extracted_data[key] = value
    return extracted_data


def create_docs(cv_files): 

    # List to store extracted data from each document
    extracted_data_list = []

    # Loop through each PDF file
    for cv_file in cv_files:
        # Initialize PDF reader for each file
        pdfreader = PdfReader(cv_file)

        # Read text from the current PDF file
        raw_text = ''
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text += content

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Update FAISS with the current document
        embeddings = OpenAIEmbeddings()
        document_search = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        # User query for CV data extraction
        query = "Extract all the following values: Full Name, Email, Phone Number, Key Skills, Education, Location, Current Company, Current Designation, and Total Years of Experience from the PDF.Look for the name at the beggining of the document. Years of experience should be numeric and not tagged with any text as it has to be stored in a numeric field"

        # Perform similarity search and run QA chain on the selected document
        docs = document_search.similarity_search(query)
        result = chain.run(input_documents=docs, question=query)

        extracted_data = parse_result_string(result)

        # Create a dictionary for the extracted data with keys as labels
        extracted_data_dict = {
            "Full Name": extracted_data.get("Full Name", ""),
            "Email": extracted_data.get("Email", ""),
            "Phone Number": extracted_data.get("Phone Number", ""),
            "Key Skills": extracted_data.get("Key Skills", ""),
            "Education": extracted_data.get("Education", ""),
            "Location": extracted_data.get("Location", ""),
            "Current Company": extracted_data.get("Current Company", ""),
            "Current Designation": extracted_data.get("Current Designation", ""),
            "Total Years of Experience": extracted_data.get("Total Years of Experience", "")
        }

        # Append extracted data to the list
        extracted_data_list.append(extracted_data_dict)

    # Convert the list of dictionaries to JSON format
    json_data = json.dumps(extracted_data_list, indent=2)

    return json_data




