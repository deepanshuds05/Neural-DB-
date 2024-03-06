from thirdai import neural_db as ndb
from langchain_openai import AzureChatOpenAI
from paperqa.prompts import qa_prompt
from paperqa.chains import make_chain
import pandas as pd
import fitz
from langchain.text_splitter import CharacterTextSplitter
# from langchain.prompts import PromptTemplate
from thirdai import licensing, neural_db as ndb
# import tqdm
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize
import os

 
from thirdai.neural_db import NeuralDB
if "THIRDAI_KEY" in os.environ:
    licensing.activate(os.environ["THIRDAI_KEY"])
else:
    licensing.activate("B9D043-991069-E0BBE9-9EEDE3-CB8E86-V3")  # Enter your ThirdAI key here


# Create an instance of AzureChatOpenAI
OPENAI_API_KEY = "e9d38fe1d9904e7db5222a70e4d397b8"
OPENAI_DEPLOYMENT_ENDPOINT = "https://bfslabopenai.openai.azure.com/"
OPENAI_DEPLOYMENT_NAME = "BFSLAB"
OPENAI_MODEL_NAME = "gpt-35-turbo"
OPENAI_DEPLOYMENT_VERSION = "2023-05-15"

def model():
    return AzureChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0,
        azure_endpoint=OPENAI_DEPLOYMENT_ENDPOINT,
        openai_api_version=OPENAI_DEPLOYMENT_VERSION,
        openai_api_key=OPENAI_API_KEY,
        azure_deployment=OPENAI_DEPLOYMENT_NAME
)

db30 = NeuralDB("Mar5.ndb")
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def save_chunks_to_csv(chunks, csv_path):
    df = pd.DataFrame(chunks, columns=['Text'])
    df.to_csv(csv_path, index=False)


# Load question-answer pairs from a CSV file
def get_db_model():
    db30 = ndb.NeuralDB("Mar5.ndb")
    pdf_paths = ['JPMorgan.pdf','Citi.pdf']
    # pdf_paths = ['wealth.pdf']

# pdf_paths = ['Credit Agreement Pdf.pdf']
    csv_files = []
    for pdf_path in pdf_paths:
        csv_out_path = pdf_path.split(".")[0] + ".csv"
        csv_files.append(csv_out_path)
        chunk_size = 1000
        chunk_overlap = 100
        text = extract_text_from_pdf(pdf_path)
        splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap,separator='\n')

        chunks = list(map(lambda x: x.page_content, splitter.create_documents([text])))
        print(len(chunks))
        save_chunks_to_csv(chunks, csv_out_path)
        print("Saved CSV file:", csv_out_path)

        csv_files =  ['JPMorgan.csv','Citi.csv']
        # csv_files =  ['Wealth.csv']


        csv_docs = [ndb.CSV(path=csv_file, strong_columns=['Text'], weak_columns=[], reference_columns=['Text']) for csv_file in csv_files]
        db30 = ndb.NeuralDB()
        db30.insert(csv_docs)
        # question_df = pd.read_csv("citi_vs_jp_qna.csv")
        
        for csv_file in csv_files:

            df = pd.read_csv("citi_vs_jp_qna.csv")
            # df = pd.read_csv("wealth_question_answer_pairs.csv")

            print(csv_file)
            questions = df['question'].tolist()
            answers = df['answer'].tolist()
            for question, answer in zip(questions, answers):
                db30.associate(question, answer)

        # db30.save("latest_model_UI.ndb")
    return db30


db30 = get_db_model()


def get_references(query, radius=None):
    search_results = db30.search(query, top_k=5)
    references = []
 
    for result in search_results:
        reference_text = result.context(radius=radius) if radius else result.text
        source = result.metadata.get('source', '')  # Get source or an empty string if source is not available
        # Truncate text to 200 words
        reference_text_truncated = ' '.join(word_tokenize(reference_text)[:200])
        references.append({'text': reference_text_truncated, 'source': source})
 
    return references
 

def get_answer(query, references):
    # Extract text and source separately
    references_text = [ref['text'] for ref in references]
    references_source = [ref['source'] for ref in references]
    
    # Truncate text to 200 words
    references_text_truncated = [' '.join(word_tokenize(text)[:200]) for text in references_text]
    
    # Use the default qa_prompt
    print("Query:", query)
    qa_chain = make_chain(prompt=qa_prompt, llm=model())
    
    return qa_chain.run(question=query, context='\n\n'.join(references_text_truncated), answer_length="about 100 words", source='\n\n'.join(references_source))
