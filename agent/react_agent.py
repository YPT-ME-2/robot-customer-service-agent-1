from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
# from langchain_core.messages import AIMessage
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
# from langgraph.checkpoint.memory import InMemorySaver
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from utils.path_tool import get_abs_path
from uuid import uuid4


class ReactAgent:
    def __init__(self):
        # self.memory=InMemorySaver()
        state_dir=get_abs_path("state")
        os.makedirs(state_dir,exist_ok=True)
        db_path=os.path.join(state_dir,"chat_history.sqlite3")
        self.connection=sqlite3.connect(
            db_path,
            check_same_thread=False,
        )
        self.memory = SqliteSaver(self.connection)
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
            checkpointer=self.memory
        )

    def execute_stream(self, query: str,session_id:str|None=None,):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(
            input_dict, 
            config={
                "configurable":{
                    "thread_id":session_id or str(uuid4())
                }
            },
            stream_mode="values", 
            context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"

    def execute(self,query:str,session_id:str|None=None)->str:
        result=self.agent.invoke(
            {
                "messages":[
                    {"role":"user","content":query}
                ]
            },
            config={
                "configurable":{
                    "thread_id":session_id or str(uuid4())
                }
            },
            context={"report":False},
        )
        messages=result["messages"]
        if not messages:
            raise RuntimeError("Agent 未返回消息")
        final_message=messages[-1]
        if not isinstance(final_message,AIMessage):
            raise RuntimeError("Agent 最后一条消息不是模型回答")
        if final_message.tool_calls:
            raise RuntimeError("Agent 最后一条消息仍在请求工具调用")
        return StrOutputParser().invoke(final_message).strip()

    def get_history(self,session_id:str)->list[dict]:
        snapshot=self.agent.get_state(
            {"configurable": {"thread_id": session_id}}
        )
        messages=snapshot.values.get("messages",[])
        history=[]
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                if message.tool_calls:
                    continue
                role = "assistant"
            else:
                continue
            if message.content:
                history.append({
                    "role": role,
                    "content": message.content,
                })
        return history
if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
