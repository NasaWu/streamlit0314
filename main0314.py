import pandas as pd
import json
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from modelscope import snapshot_download
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import gradio as gr
import re

####################################################################################
# D:/Programe/AIGC/0312/
loader = CSVLoader(file_path='test0312question.csv', encoding='gbk')
documents = loader.load()
df = pd.read_csv('test0312.csv', encoding='gbk')
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
model_dir = snapshot_download("AI-ModelScope/bge-large-zh-v1.5", revision='master')
embedding_path=model_dir
embeddings = HuggingFaceBgeEmbeddings(model_name = embedding_path)

vectorstore = FAISS.from_documents(
    docs,
    embedding= embeddings
)
retriever = vectorstore.as_retriever()

# def qa_system(query_vector):
#     top_docs = retriever.invoke(query_vector,top_k=3)

#     i = 0
#     formatted_data = []
#     indexlist = []
#     for i in range(3):
#         content_lines = top_docs[i].page_content.split('\n')
#         # 初始化index变量
#         index = None
#         # 遍历每一行，找到index所在的行
#         for line in content_lines:
#             if line.startswith('index:'):
#                 # 提取index后面的数字部分
#                 index = int(line.split(':')[1].strip())
#                 indexlist.append(index)  # 直接使用append方法
#         selected_rows = df.iloc[indexlist].reset_index(drop=True)  # 使用indexlist来索引行
        
#     for index, row in selected_rows.iterrows():
#         # 每行数据格式化为{"question": question, "answer": answer}的形式
#         formatted_data.append({"question": row["question"].replace("\n", "<br>"), "answer": row["answer"].replace("\n", "<br>")})
#         # 将列表转换为JSON格式的字符串，应该在循环外部执行
#           # 简化i的增加方式
#     # 循环结束后，将formatted_data转换为JSON字符串
#     json_result = json.dumps(formatted_data, ensure_ascii=False)
#     data = json.loads(json_result)
# # 创建一个新的列表来存储格式化后的结果
#     formatted_result = []
#     for item in data:
#         formatted_item = {
#             "Q": f'<span style="color: red;">{item["question"].replace("\n", "<br>")}</span>',
#             "A": f'<span style="color: green;">{item["answer"].replace("\n", "<br>")}</span>'
#         }
#         formatted_result.append(formatted_item)
#     return formatted_result
def retrieve_top_docs(query_vector, top_k=2):
    return retriever.invoke(query_vector, top_k=top_k)

def extract_indices(top_docs):
    indexlist = []
    for doc in top_docs:
        content_lines = doc.page_content.split('\n')
        for line in content_lines:
            if line.startswith('index:'):
                index = int(line.split(':')[1].strip())
                indexlist.append(index)
    return indexlist

def get_selected_rows(df, indexlist):
    return df.iloc[indexlist].reset_index(drop=True)

def format_data(selected_rows):
    formatted_data = []
    for index, row in selected_rows.iterrows():
        formatted_data.append({
            "question": row["question"].replace("\n", "<br>"),
            "answer": row["answer"].replace("\n", "<br>")
        })
    return formatted_data

def format_json_question(formatted_data):
    formatted_question = []
    for item in formatted_data:
        formatted_q = {
            "Q": f'<span style="color: red;">{item["question"]}</span>'
            # "A": f'<span style="color: green;">{item["answer"]}</span>'
        }
        formatted_question.append(formatted_q)
    return formatted_question

def format_json_answer(formatted_data):
    formatted_answer = []
    for item in formatted_data:
        formatted_a = {
            # "Q": f'<span style="color: red;">{item["question"]}</span>'
            "A": f'<span style="color: green;">{item["answer"]}</span>'
        }
        formatted_answer.append(formatted_a)
    return formatted_answer

def format_json_result(formatted_data):
    formatted_result = []
    for item in formatted_data:
        formatted_item = {
            "Q": f'<span style="color: red;">{item["question"]}</span>',
            "A": f'<span style="color: green;">{item["answer"]}</span>'
        }
        formatted_result.append(formatted_item)
    return formatted_result

def qa_system(query_vector, df):
    top_docs = retrieve_top_docs(query_vector)
    indexlist = extract_indices(top_docs)
    selected_rows = get_selected_rows(df, indexlist)
    formatted_data = format_data(selected_rows)
    formatted_question = format_json_question(formatted_data)
    formatted_answer = format_json_answer(formatted_data)
    formatted_result = format_json_result(formatted_data)
    return formatted_result


#########################
# ############# streamlit 
# ########################
import streamlit as st

st.title("IPS AMP Chat")
# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []
# 显示已有的聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# 聊天输入框
if prompt := st.chat_input("What is up?"):
    top_docs = retrieve_top_docs(prompt)
    indices= extract_indices(top_docs)
    answer = get_selected_rows(df,indices)
    # 将用户的输入添加到聊天记录中
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # 在这里调用您自己的逻辑来获取回答
    # 假设您的逻辑是一个名为 get_answer 的函数  
    # 显示助手的回答
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.table(answer[['question', 'answer']])
# 定义您的逻辑函数
# def get_answer(user_input):
#     # 这里是您的逻辑，返回一个字符串作为回答
#     # 例如，我们这里简单地回复 "Hello World!"
#     return "Hello World!"


