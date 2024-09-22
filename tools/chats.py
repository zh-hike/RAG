import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.rag.llms import GLM4
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def main():
    model_name = "ZhipuAI/glm-4-9b-chat"
    gen_kwargs = {"max_length": 10000, "do_sample": True, "top_k": 3}
    model = GLM4(
        model_name,
        gen_kwargs=gen_kwargs
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个善于聊天的机器人，你的任务就是用聊天使我开心。")
    ])

    
    while True:
        inputs = input("请输入： ")
        prompt.append(("user", inputs))
        if inputs == "quit":
            break

        chain = prompt | model
        res = ""
        for chunk in chain.stream({}):
            content = chunk.content
            res += content
            print(content, end="")
        print()
        prompt.append(("assistant", res))


if __name__ == "__main__":
    main()