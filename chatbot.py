
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from flask import *
import os
app = Flask(__name__)



llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key='AIzaSyDClctnkjwy58BexpYdwyDIII81_UG-95M')
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyDClctnkjwy58BexpYdwyDIII81_UG-95M')
data_folder = 'D:/Private/CODE/Chat_bot/data'
vector_db_path = 'D:/Private/CODE/Chat_bot/vectorstores/db_faiss'

@app.route('/')
def welcome():
    return render_template('welcome.html')


def load_text():
    loader = TextLoader("D:/Private/CODE/Chat_bot/data/result.txt", encoding="utf-8")
    text_splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=1024,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    documents = loader.load()
    texts = text_splitter.split_documents(documents)

    return texts

folder_path = 'D:/Private/CODE/Chat_bot/data'

def create_db():
    # for filename in os.listdir(folder_path):
    #     if filename.endswith(".txt"):
    #         file_path = os.path.join(folder_path, filename)
    #         with open(file_path, 'r', encoding='utf-8') as f:
    #             raw_text = f.read()
    #             text_splitter = CharacterTextSplitter(
    #                 separator="\n",
    #                 chunk_size=1024,
    #                 chunk_overlap=50,
    #                 length_function=len

                # )
    
    raw_text = ''
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            print(file_path)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                raw_text += text + '\n'

    raw_text
    # with open('D:/Private/CODE/Chat_bot/data/data.txt','r',encoding='utf-8') as f:
    #     raw_text = f.read()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1024,
        chunk_overlap=50,
        length_function=len

    )

    chunks = text_splitter.split_text(raw_text)

    # Embeding
    embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"trust_remote_code":True})


    # Dua vao Faiss Vector DB
    db = FAISS.from_texts(texts=chunks, embedding=embeddings)
    db.save_local(vector_db_path)
    return db
 

def read_vectors():
    embeddings = SentenceTransformerEmbeddings(model_name="keepitreal/vietnamese-sbert", model_kwargs={"trust_remote_code":True})
    db = FAISS.load_local(vector_db_path,embeddings,allow_dangerous_deserialization=True)

    return db


def create_template():
    template = """Sử dụng thông tin sau đây để trả lời câu hỏi. Chỉnh sửa hoặc diễn giải lại thông tin một cách ngắn gọn. Nếu bạn không biết câu trả lời, hãy nói 'Tôi không biết'.

    Thông tin:
    {context}

    Câu hỏi:
    {question}
    Trả lời dựa trên dữ liệu:"""
    return template

def set_custom_prompt(template):
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=template,
                            input_variables=['context', 'question'])
    return prompt


def create_chain(db,prompt):
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={'prompt': prompt}
    )
    return chain



