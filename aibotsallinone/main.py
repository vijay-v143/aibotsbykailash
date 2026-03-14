
import os
import anthropic
import openai
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Multi-AI Chat API")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────

class Message(BaseModel):
    role: str        # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    ai: str          # "claude", "chatgpt", or "gemini"
    history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    reply: str
    ai: str


# ─────────────────────────────────────────
#  Health Check
# ─────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "✅ Multi-AI Chat backend is running!"}


# ─────────────────────────────────────────
#  Main Chat Route
# ─────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.ai == "claude":
        return chat_claude(req)
    elif req.ai == "chatgpt":
        return chat_chatgpt(req)
    elif req.ai == "gemini":
        return chat_gemini(req)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown AI: '{req.ai}'. Choose 'claude', 'chatgpt', or 'gemini'."
        )


# ─────────────────────────────────────────
#  Claude (Anthropic)
# ─────────────────────────────────────────

def chat_claude(req: ChatRequest):
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY is missing from your .env file."
        )

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Build message history
        messages = [{"role": m.role, "content": m.content} for m in req.history]
        messages.append({"role": "user", "content": req.message})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=messages
        )

        return ChatResponse(reply=response.content[0].text, ai="claude")

    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid Anthropic API key.")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Anthropic rate limit reached. Try again later.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude error: {str(e)}")


# ─────────────────────────────────────────
#  ChatGPT (OpenAI)
# ─────────────────────────────────────────

def chat_chatgpt(req: ChatRequest):
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing from your .env file."
        )

    try:
        client = openai.OpenAI(api_key=api_key)

        # Build message history
        messages = [{"role": m.role, "content": m.content} for m in req.history]
        messages.append({"role": "user", "content": req.message})

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=messages
        )

        return ChatResponse(reply=response.choices[0].message.content, ai="chatgpt")

    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key.")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI rate limit reached. Try again later.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChatGPT error: {str(e)}")


# ─────────────────────────────────────────
#  Gemini (Google)
# ─────────────────────────────────────────

def chat_gemini(req: ChatRequest):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing from your .env file."
        )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Gemini uses "model" instead of "assistant"
        history = []
        for m in req.history:
            role = "model" if m.role == "assistant" else "user"
            history.append({"role": role, "parts": [m.content]})

        chat_session = model.start_chat(history=history)
        response = chat_session.send_message(req.message)

        return ChatResponse(reply=response.text, ai="gemini")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")
