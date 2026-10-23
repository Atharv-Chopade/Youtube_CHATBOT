# YouTube AI Chatbot Frontend

This frontend is designed for your existing `youtube_rag_langchain.py` RAG functions.

## Files
- `index.html` — page structure
- `style.css` — responsive dark UI
- `app.js` — video loading and chat requests

## Important: backend required
Your current Python file is a **Streamlit app**, not a REST API. The frontend calls these endpoints on the same origin:
- `POST /api/load-video` with `{ "youtube_url": "...", "api_key": "...", "model_name": "..." }`
  - expected response: `{ "video_id": "...", "chunk_count": 12 }`
- `POST /api/chat` with `{ "video_id": "...", "question": "..." }`
  - expected response: `{ "answer": "..." }`
- Optional `POST /api/reset` with `{ "video_id": "..." }`

A small Flask/FastAPI backend must expose these endpoints and call the functions in `youtube_rag_langchain.py`. Do not open `index.html` directly and expect the AI to work before those endpoints exist.

## Run the frontend locally
For visual preview, open `index.html` in your browser. The UI will render, but API actions need the backend.
For integrated use, serve the frontend files from the same backend origin, or change `API_BASE` in `app.js` to your backend URL and configure CORS.

## Keep secrets safe
For local development, setting `GROQ_API_KEY` in your backend `.env` is safer than putting it in the browser settings. Never publish a real API key.
