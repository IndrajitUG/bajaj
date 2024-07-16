# # import os
# # import streamlit as st
# # from langchain.embeddings.openai import OpenAIEmbeddings
# # from langchain_openai import ChatOpenAI
# # from langchain.chains import ConversationalRetrievalChain
# # from langchain.memory import ConversationBufferMemory
# # from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
# # from PIL import Image

# # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # system_template = r'''
# # Use the following context to answer the questions.
# # If the answer is not available in the context your respone should be : "Not Found"
# # ---------------
# # Context: ```{context}```
# # '''
# # user_template = '''
# # Question: ```{question}```
# # '''

# # messages = [
# #     SystemMessagePromptTemplate.from_template(system_template),
# #     HumanMessagePromptTemplate.from_template(user_template)
# # ]

# # qa_prompt = ChatPromptTemplate.from_messages(messages)

# # def load_documents(file):
# #     import os
# #     name,extension = os.path.splitext(file)
    
# #     if extension == ".pdf":
# #         from langchain.document_loaders import PyPDFLoader
# #         print(f'Loading {file}')
# #         loader = PyPDFLoader(file)
# #         #loader = PyPDFLoader("url")
# #     elif extension == ".docx":
# #         from langchain.document_loader import Docx2txtLoader
# #         print(f'Loading {file}')
# #         loader = Docx2txtLoader(file)
# #     elif extension == '.txt':
# #         from langchain.document_loaders import TextLoader
# #         loader = TextLoader(file)
# #     else:
# #         print("Document format not supported")
# #         return None
        
# #     data = loader.load()
# #     return data

# # def chunk_data(data,chunk_size=256,chunk_overlap =20):
# #     from langchain.text_splitter import RecursiveCharacterTextSplitter
# #     text_splitter = RecursiveCharacterTextSplitter(
# #         chunk_size = chunk_size,
# #         chunk_overlap = chunk_overlap
# #     )
# #     chunks = text_splitter.split_documents(data)
# #     return chunks

# # def create_embedding_pinecone(chunks, index_name="pdf"):
# #     import pinecone
# #     from langchain_community.vectorstores import Pinecone
# #     from langchain_openai import OpenAIEmbeddings
# #     from pinecone import PodSpec
    
# #     embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    
# #     # Initialize Pinecone
# #     pc = pinecone.Pinecone()
    
# #     # Create Pinecone index
# #     if index_name not in pc.list_indexes().names():
# #         pc.create_index(
# #             name=index_name,
# #             dimension=1536,
# #             metric="cosine",
# #             spec=PodSpec(
# #                 environment='gcp-starter'
# #             )
# #         )
    
# #     vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
# #     return vector_store

# # def load_embeddings_pinecone(index_name="pdf"):
# #     from langchain.vectorstores import Pinecone
# #     from langchain_openai import OpenAIEmbeddings
    
# #     embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    
# #     # Initialize Pinecone 
# #     vector_store = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
# #     return vector_store

# # if __name__ == "__main__":
# #     from dotenv import load_dotenv, find_dotenv
# #     load_dotenv(find_dotenv(), override=True)

# #     llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

# #     if 'vs' not in st.session_state:
# #         st.session_state['vs'] = None

# #     st.header("GPT")
# #     # st.image(logo, width=80)

# #     with st.sidebar:
# #         uploaded_file = st.file_uploader('Upload a file: ',type=['pdf','docs','txt'])
        
# #         if uploaded_file:
# #             bytes_data = uploaded_file.read()
# #             file_name = os.path.join('./',uploaded_file.name)
# #             with open(file_name,'wb') as f:
# #                 f.write(bytes_data)
# #             data = load_documents(uploaded_file)
# #             chunks = chunk_data(data)
# #             vector_store = create_embedding_pinecone(chunks)
# #             st.session_state['vs'] = vector_store
# #     st.text_area('Suggestions:', value="Try: Give me 5 latest horror scripts or Scripts similar to Fast and furious\n or\n Annabelle", height=100)
# #     q = st.text_input("Enter the question")
    
# #     if q:
# #         with st.spinner("Running..."):
# #             if st.session_state['vs'] is not None:
# #                 vector_store = st.session_state['vs']
# #             else:
# #                 vector_store = load_embeddings_pinecone()
# #                 st.session_state['vs'] = vector_store

# #             retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 20})
# #             crc = ConversationalRetrievalChain.from_llm(
# #                 llm=llm,
# #                 retriever=retriever,
# #                 memory=memory,
# #                 chain_type="stuff",
# #                 combine_docs_chain_kwargs={'prompt': qa_prompt},
# #                 verbose=False
# #             )
# #             answer = ask_question(q, crc)
# #         st.text_area('Answer:', value=answer, height=300)

# #         st.divider()
# #         if 'history' not in st.session_state:
# #             st.session_state.history = ''
# #         value = f'Q: {q} \nA: {answer}'
# #         st.session_state.history = f'{value} \n {"-"*100}\n {st.session_state.history}'
# #         h = st.session_state.history
# #         st.text_area(label="Chat history",value=h, key='history',height = 300)

# import os
# import streamlit as st
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
# from PIL import Image

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# system_template = r'''
# Use the following context to answer the questions.
# If the answer is not available in the context your respone should be : "Not Found"
# ---------------
# Context: ```{context}```
# '''
# user_template = '''
# Question: ```{question}```
# '''

# messages = [
#     SystemMessagePromptTemplate.from_template(system_template),
#     HumanMessagePromptTemplate.from_template(user_template)
# ]

# qa_prompt = ChatPromptTemplate.from_messages(messages)

# def load_documents(uploaded_file):
#     import os
#     from io import BytesIO
    
#     file_name = uploaded_file.name
#     name, extension = os.path.splitext(file_name)
    
#     file_content = BytesIO(uploaded_file.read())
    
#     if extension == ".pdf":
#         from langchain.document_loaders import PyPDFLoader
#         print(f'Loading {file_name}')
#         loader = PyPDFLoader(file_content)
#     elif extension == ".docx":
#         from langchain.document_loaders import Docx2txtLoader
#         print(f'Loading {file_name}')
#         loader = Docx2txtLoader(file_content)
#     elif extension == '.txt':
#         from langchain.document_loaders import TextLoader
#         loader = TextLoader(file_content)
#     else:
#         print("Document format not supported")
#         return None
        
#     data = loader.load()
#     return data

# def chunk_data(data, chunk_size=256, chunk_overlap=20):
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
#     chunks = text_splitter.split_documents(data)
#     return chunks

# def create_embedding_pinecone(chunks, index_name="pdf"):
#     import pinecone
#     from langchain_community.vectorstores import Pinecone
#     from langchain_openai import OpenAIEmbeddings
#     from pinecone import PodSpec
    
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    
#     # Initialize Pinecone
#     pc = pinecone.Pinecone()
    
#     # Create Pinecone index
#     if index_name not in pc.list_indexes().names():
#         pc.create_index(
#             name=index_name,
#             dimension=1536,
#             metric="cosine",
#             spec=PodSpec(
#                 environment='gcp-starter'
#             )
#         )
    
#     vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
#     return vector_store

# def load_embeddings_pinecone(index_name="pdf"):
#     from langchain.vectorstores import Pinecone
#     from langchain_openai import OpenAIEmbeddings
    
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    
#     # Initialize Pinecone 
#     vector_store = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
#     return vector_store

# if __name__ == "__main__":
#     from dotenv import load_dotenv, find_dotenv
#     load_dotenv(find_dotenv(), override=True)

#     llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

#     if 'vs' not in st.session_state:
#         st.session_state['vs'] = None

#     st.header("GPT")
#     # st.image(logo, width=80)

#     with st.sidebar:
#         uploaded_file = st.file_uploader('Upload a file: ', type=['pdf', 'docx', 'txt'])
        
#         if uploaded_file:
#             data = load_documents(uploaded_file)
#             if data is not None:
#                 chunks = chunk_data(data)
#                 vector_store = create_embedding_pinecone(chunks)
#                 st.session_state['vs'] = vector_store
#     st.text_area('Suggestions:', value="Try: Give me 5 latest horror scripts or Scripts similar to Fast and furious\n or\n Annabelle", height=100)
#     q = st.text_input("Enter the question")
    
#     if q:
#         with st.spinner("Running..."):
#             if st.session_state['vs'] is not None:
#                 vector_store = st.session_state['vs']
#             else:
#                 vector_store = load_embeddings_pinecone()
#                 st.session_state['vs'] = vector_store

#             retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 20})
#             crc = ConversationalRetrievalChain.from_llm(
#                 llm=llm,
#                 retriever=retriever,
#                 memory=memory,
#                 chain_type="stuff",
#                 combine_docs_chain_kwargs={'prompt': qa_prompt},
#                 verbose=False
#             )
#             answer = ask_question(q, crc)
#         st.text_area('Answer:', value=answer, height=300)

#         st.divider()
#         if 'history' not in st.session_state:
#             st.session_state.history = ''
#         value = f'Q: {q} \nA: {answer}'
#         st.session_state.history = f'{value} \n {"-"*100}\n {st.session_state.history}'
#         h = st.session_state.history
#         st.text_area(label="Chat history", value=h, key='history', height=300)

import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from PIL import Image

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

system_template = r'''
Use the following context to answer the questions.
Give answer in points format.
If the answer is not available in the context your response should be : "Not Found"
---------------
Context: ```{context}```
'''
user_template = '''
Question: ```{question}```
'''

logo_path = "./bajaj.png"
logo_url = "https://w7.pngwing.com/pngs/552/21/png-transparent-bajaj-auto-logo-motorcycle-company-company-logo-blue-text-trademark.png"

if os.path.exists(logo_path):
    logo = Image.open(logo_path)
else:
    logo = logo_url

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template)
]

qa_prompt = ChatPromptTemplate.from_messages(messages)

def save_uploaded_file(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_file.name

def load_documents(file_path):
    import os
    
    name, extension = os.path.splitext(file_path)
    
    if extension == ".pdf":
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file_path}')
        loader = PyPDFLoader(file_path)
    elif extension == ".docx":
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file_path}')
        loader = Docx2txtLoader(file_path)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file_path)
    else:
        print("Document format not supported")
        return None
        
    data = loader.load()
    return data

def chunk_data(data, chunk_size=512, chunk_overlap=40):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embedding_pinecone(chunks, index_name="pdf"):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    
    # Initialize Pinecone
    pc = pinecone.Pinecone()
    pc.delete_index(index_name)
    # Create Pinecone index
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=PodSpec(
                environment='gcp-starter'
            )
        )
    
    vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    return vector_store

def load_embeddings_pinecone(index_name="pdf"):
    from langchain.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    
    # Initialize Pinecone 
    vector_store = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    return vector_store

def ask_question(q, chain):
    result = chain.invoke({'question': q})
    return result['answer']

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

    if 'vs' not in st.session_state:
        st.session_state['vs'] = None

    st.image(logo, width=130)
    st.header("Document Analysis GPT")

    with st.sidebar:
        uploaded_file = st.file_uploader('Upload a file: ', type=['pdf', 'docx', 'txt'])
        add_data = st.button("Add Data",on_click= clear_history)

        if uploaded_file and add_data:
            with st.spinner("Reading and embedding file..."):
                file_path = save_uploaded_file(uploaded_file)
                data = load_documents(file_path)
                if data is not None:
                    chunks = chunk_data(data)
                    vector_store = create_embedding_pinecone(chunks)
                    st.session_state['vs'] = vector_store
                st.success('File uploaded and embedded successfully...')

    q = st.text_input("Enter the question")
    submit = st.button("Submit")
    if q and submit:
        with st.spinner("Running..."):
            if st.session_state['vs'] is not None:
                vector_store = st.session_state['vs']
            else:
                vector_store = load_embeddings_pinecone()
                st.session_state['vs'] = vector_store

            retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 20})
            crc = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                chain_type="stuff",
                combine_docs_chain_kwargs={'prompt': qa_prompt},
                verbose=False
            )
            answer = ask_question(q, crc)
        st.text_area('Answer:', value=answer, height=300)

        st.divider()
        if 'history' not in st.session_state:
            st.session_state.history = ''
        value = f'Q: {q} \nA: {answer}'
        st.session_state.history = f'{value} \n {"-"*100}\n {st.session_state.history}'
        h = st.session_state.history
        st.text_area(label="Chat history", value=h, key='history', height=300)

