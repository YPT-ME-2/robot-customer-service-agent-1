from fastapi import FastAPI,HTTPException,Path
from pydantic import BaseModel, Field
from utils.logger_handler import logger
app = FastAPI()
from agent.react_agent import ReactAgent
from fastapi.responses import StreamingResponse

agent = ReactAgent()

@app.get("/hello")
def hello(name: str = "同学"):
    return {"message": f"你好，{name}"}

class ChatRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    question: str = Field(min_length=1, max_length=2000)

@app.post("/chat")
def chat(request: ChatRequest):
    question=request.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail="问题不能只包含空格或换行"
        )
    answer = ""
    try:
        # for text in agent.execute_stream(question):
        #     answer = text.strip()
        answer=agent.execute(question,session_id=request.session_id,)
    except Exception:
        logger.exception("聊天接口执行失败")
        raise HTTPException(
            status_code=500,
            detail="暂时无法生成回答，请检查服务状态后重试",
        )
    if not answer:
        raise HTTPException(
            status_code=502,
            detail="未获得有效回答，请重试",
        )

    return {
        "question": question,
        "answer": answer,
    }

@app.post("/chat/stream")
def chat_stream(request:ChatRequest):
    question=request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="问题不能只包含空格或换行",
        )
    return StreamingResponse(
        agent.execute_stream(question,session_id=request.session_id),
        media_type="text/plain; charset=utf-8",
    )

@app.get("/sessions/{session_id}/messages")
def get_session_messages(
    session_id:str=Path(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
):
    messages=agent.get_history(session_id)
    return {
        "session_id": session_id,
        "messages": messages,
    }