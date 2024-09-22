import os
from typing import Any, Dict, List, Optional
from rich import print
from threading import Thread
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import SimpleChatModel
from langchain_core.messages import BaseMessage, SystemMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from modelscope import snapshot_download

__DIR__ = os.path.dirname( os.path.abspath(__file__))


class GLM4(SimpleChatModel):

    model: AutoModelForCausalLM = None
    tokenizer: AutoTokenizer = None
    gen_kwargs: dict = None

    def __init__(
        self,
        model_name: str,
        gen_kwargs: dict = {"max_length": 10000, "do_sample": True, "top_k": 3},
        **kwargs,
    ):
        super().__init__()
        
        __model_path__ = os.path.abspath(os.path.join(__DIR__, "../../../../models"))
        if not os.path.exists(os.path.join(__model_path__, model_name)):
            print(f"未发现模型 {os.path.join(__model_path__, model_name)}，正在下载...")
            snapshot_download(model_name, cache_dir=__model_path__)

        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.join(__model_path__, model_name), trust_remote_code=True, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(__model_path__, model_name), trust_remote_code=True
        )
        self.gen_kwargs = gen_kwargs

    def _call(
        self,
        prompt: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ):
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, **self.gen_kwargs)
        outputs = outputs[:, inputs.input_ids.shape[1] :]
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return response

    def _stream(
        self,
        prompt: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ):

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        generation_kwargs = dict(inputs, **self.gen_kwargs, streamer=streamer)

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for char in streamer:
            chunk = SystemMessageChunk(content=char)
            yield ChatGenerationChunk(message=chunk, generation_info={})

    @property
    def _llm_type(self) -> str:
        return "glm-4-9b-chat"


if __name__ == "__main__":
    model_name = "ZhipuAI/glm-4-9b-chat"
    gen_kwargs = {"max_length": 10000, "do_sample": True, "top_k": 3}
    model = GLM4(
        model_name,
        gen_kwargs=gen_kwargs
    )


    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个故事大王，我给你一个主题，你的任务就是给我讲关于这个主题的{theme}故事"),
        ("user", "这个主题是{title}")
    ])

    chain = prompt | model

    response = chain.invoke({"theme": "搞笑", "title": "猪八戒"})

    print(f"response: {response.content}")

    for chunk in chain.stream({"theme": "恐怖", "title": "手机"}):
        print(chunk.content, end="")