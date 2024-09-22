# RAG Demo

> 作者：zhhike\
> 邮箱：zhhike@qq.com\
> 主页：[http://github.com/zh-hike](http://github.com/zh-hike)

## 一、环境配置
环境配置比较简单，首先根据你的需求选择CPU或者GPU配置
### 1. CPU
```
peotry install
```

### 2. GPU
GPU 需要有 cuda11.2+。
运行以下命令安装环境
```
poetry install
```

## 二、使用方法
### 1. 多轮对话
这里使用 *glm4* 举例。
```
poetry run python tools/chats.py
```