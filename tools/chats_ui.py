import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.rag.llms import GLM4
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from rich import print
import streamlit as st


print("加载模型")

model_name = "ZhipuAI/glm-4-9b-chat"
gen_kwargs = {"max_length": 10000, "do_sample": True, "top_k": 3}



model = GLM4(
    model_name,
    gen_kwargs=gen_kwargs
)


print("加载完成")

st.title("💬 聊天机器人")



prompts = ChatPromptTemplate.from_messages([
    ("system", "你是一个善于聊天的机器人，你的任务就是用聊天使我开心。")
])


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "说出你的诉求?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    prompts.append(("user", prompt))
    chain = prompts | model

    response = chain.invoke({})

    msg = response.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

    prompts.append(("assistant", msg))
