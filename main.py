# 📋 main.py · [실습 6-1] 백엔드 — 사진 · 음성 도우미 API (Vercel 에 올릴 main.py · POST 한 번에 요청 1번)
# · 키는 서버 환경 변수 GEMINI_API_KEY · 접근 코드는 APP_ACCESS_CODE (13주차와 같음)
# · 사진 · 음성은 화면(Streamlit)이 base64 글자로 바꿔 JSON 으로 보냄 — Vercel 요청 본문 4.5MB 이하
import base64
import binascii
import os
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.5-flash-lite"
MAX_B64 = 4_000_000                 # base64 글자 수 상한 — Vercel 요청 본문 4.5MB 보다 조금 작게
VOICE_PROMPTS = {
    "받아쓰기": "이 음성을 한국어로 그대로 받아써 줘. 잘 안 들리는 부분은 (?) 로 표시해 줘.",
    "한 줄 요약": "이 음성의 내용을 한 문장으로 요약해 줘.",
    "영어로 번역": "이 음성의 내용을 자연스러운 영어로 번역해 줘.",
}
app = FastAPI(title="사진 · 음성 도우미 API", description="사진이나 음성을 base64 로 보내면 Gemini 가 읽고 답합니다.")


class PhotoRequest(BaseModel):
    image_b64: str = Field(min_length=1, description="사진(JPEG · PNG)을 base64 로 바꾼 글자")
    mime_type: Literal["image/jpeg", "image/png"] = "image/jpeg"
    question: str = Field(default="이 사진에 무엇이 있는지 3문장 이내로 설명해 줘.", max_length=300)


class VoiceRequest(BaseModel):
    audio_b64: str = Field(min_length=1, description="녹음(WAV)을 base64 로 바꾼 글자")
    task: Literal["받아쓰기", "한 줄 요약", "영어로 번역"] = "받아쓰기"


class Answer(BaseModel):
    answer: str
    tokens: int | None = None


def check_access(access_code):
    """APP_ACCESS_CODE 가 정해져 있으면, 요청 헤더 X-Access-Code 가 같아야만 통과"""
    expected = os.environ.get("APP_ACCESS_CODE")
    if expected and access_code != expected:
        raise HTTPException(status_code=401, detail="접근 코드가 틀렸습니다 — 헤더 X-Access-Code 를 확인하세요")


def check_b64(text):
    """base64 글자가 맞는지 · 너무 크지 않은지 먼저 검사 — 틀리면 Gemini 까지 가지 않음"""
    if len(text) > MAX_B64:
        raise HTTPException(status_code=413, detail="파일이 너무 큽니다 — 사진은 줄이고, 녹음은 짧게 보내세요")
    try:
        base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=422, detail="base64 글자가 아닙니다 — 화면에서 b64encode 했는지 확인하세요")


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="서버에 GEMINI_API_KEY 가 없습니다 — 환경 변수를 확인하세요")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=120_000, retry_options=types.HttpRetryOptions(http_status_codes=[500, 503])),
    )


def ask(text, part):
    """글 부분 + 사진(음성) 부분을 Gemini 에 보내고 Answer 로 돌려줌"""
    client = get_client()                           # 변수에 담아 둠 (9주차 — 담지 않으면 요청 전에 닫힘)
    try:
        reply = client.interactions.create(model=MODEL_NAME, store=False, input=[{"type": "text", "text": text}, part])
    except Exception as error:
        code = getattr(error, "status_code", None)
        if code == 429:
            raise HTTPException(status_code=429, detail="Gemini 무료 한도를 넘었습니다 — 1분 뒤 다시 요청하세요")
        raise HTTPException(status_code=502, detail=f"Gemini 요청 실패 (상태 번호 {code})")
    return Answer(answer=reply.output_text, tokens=reply.usage.total_tokens)


@app.get("/")
def home():
    return {"message": "사진 · 음성 도우미 API 입니다. /docs 에서 사용법을 보세요.", "tasks": list(VOICE_PROMPTS)}


@app.post("/describe", response_model=Answer)
def describe(request: PhotoRequest, x_access_code: str | None = Header(default=None)):
    check_access(x_access_code)                     # ① 접근 코드
    check_b64(request.image_b64)                    # ② 크기 · base64 검사
    return ask(request.question, {"type": "image", "data": request.image_b64, "mime_type": request.mime_type})


@app.post("/listen", response_model=Answer)
def listen(request: VoiceRequest, x_access_code: str | None = Header(default=None)):
    check_access(x_access_code)
    check_b64(request.audio_b64)
    return ask(VOICE_PROMPTS[request.task], {"type": "audio", "data": request.audio_b64, "mime_type": "audio/wav"})
