# 🎥 YouTube RAG Chatbot

An AI-powered YouTube chatbot that allows users to ask questions about a YouTube video and receive answers based on its transcript. The project uses **Retrieval-Augmented Generation (RAG)** with LangChain and Groq AI.

##  Features

-  Load a YouTube video using its URL.
-  Extract the video's available transcript.
-  Split transcript text into smaller chunks for efficient retrieval.
-  Generate embeddings using Hugging Face.
-  Retrieve relevant transcript sections using FAISS.
-  Generate answers using Groq AI.
-  Ask questions about the video content.
-  FastAPI backend for handling requests.

##  Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| FastAPI | Backend API |
| LangChain | RAG pipeline and LLM integration |
| Groq AI | Large language model inference |
| Hugging Face | Text embeddings |
| FAISS | Vector similarity search |
| YouTube Transcript API | Fetch video transcripts |
| uv | Python dependency and environment management |

## ⚙️ How It Works

1. **Enter a YouTube URL:** Provide the link to the video you want to explore.
2. **Extract the transcript:** The application retrieves the available captions.
3. **Split the transcript:** The text is divided into smaller chunks.
4. **Create embeddings:** Hugging Face converts text chunks into numerical vectors.
5. **Store and retrieve:** FAISS searches for transcript chunks relevant to your question.
6. **Generate an answer:** Groq AI uses the retrieved context to generate a response.

## Prerequisites

Before running the project, install:

- Python 3.10 or newer
- Git
- uv
- A Groq API key

Get your API key from [Groq Console](https://console.groq.com/).

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Atharv-Chopade/Youtube_CHATBOT.git
cd Youtube_CHATBOT
```

### 2. Install uv

If uv is not installed, install it using:

```bash
pip install uv
```

Verify the installation:

```bash
uv --version
```

### 3. Install project dependencies

Install the dependencies listed in `requirements_backend.txt`:

```bash
uv pip install --system -r requirements_backend.txt
```

Alternatively, to use an isolated environment managed by uv, create and activate a virtual environment:

```bash
uv venv
.venv\Scripts\activate
uv pip install -r requirements_backend.txt
```

On Windows PowerShell, the activation command above works after creating the environment. If activation is blocked by your execution policy, use `uv run` commands instead.

## Environment Variables

Create a `.env` file in the root directory of the project and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your actual API key.

**Important:** Never upload your real API key or personal YouTube cookies to GitHub. Keep `.env` and `cookies.txt` out of version control.

## Run the Application

### Start the backend

From the project root, run:

```bash
uv run --with-requirements requirements_backend.txt uvicorn youtubeRagbackend:app --reload
```

The backend should start at:

```text
http://127.0.0.1:8000
```

Open the FastAPI documentation to explore the available endpoints:

```text
http://127.0.0.1:8000/docs
```

Keep this terminal running while using the application.

### Open the frontend

Open the `youtube_chatbot_frontend` folder and follow the frontend's configured launch method. If it contains an `index.html` file, you can use VS Code Live Server or another local HTTP server to serve it.

For example, if the frontend entry point is `index.html`, run this command from the frontend directory:

```bash
uv run python -m http.server 5500
```

Then open:

```text
http://127.0.0.1:5500
```

If your frontend uses a different entry point or development server, use its corresponding start command instead.

# How to Use

1. Start the backend and frontend.
2. Open the frontend in your browser.
3. Paste a valid YouTube video URL.
4. Load the video and wait for transcript processing.
5. Enter a question about the video.
6. Read the answer generated using the retrieved transcript content.

## RAG Architecture

```text
YouTube Video URL
        |
        v
Extract Video ID
        |
        v
Fetch Transcript
        |
        v
Split Text into Chunks
        |
        v
Hugging Face Embeddings
        |
        v
FAISS Vector Store
        |
        v
Retrieve Relevant Chunks
        |
        v
Groq AI + LangChain
        |
        v
Answer the User's Question
```

##  Troubleshooting

**1. Groq API key error**
- Check that `GROQ_API_KEY` is set correctly in `.env`.
- Restart the backend after changing environment variables.
- Verify your API key and model availability in Groq Console.

**2. No transcript available**
- Make sure the video has captions or subtitles.
- Some videos do not provide accessible transcripts.

**3. YouTube blocks transcript requests**
- YouTube may temporarily block requests from your IP address.
- Try another video or run the application from a different permitted network.
- If needed, configure the supported cookies or proxy options in the backend.

**4. Module not found**
- Install dependencies using the commands in the installation section.
- Ensure you are running the app from the repository root and using the intended Python environment.

**5. Frontend cannot connect to the backend**
- Make sure the backend is running on port `8000`.
- Check the API URL configured in the frontend.
- Check the browser console for network or CORS errors.

## Security

- Do not commit API keys, cookies, passwords, or other credentials.
- Do not expose secret keys in frontend JavaScript.
- Keep credentials in environment variables or another secure configuration mechanism.
- Use appropriate authentication and access controls before deploying the application publicly.

## Future Improvements

- Support multiple YouTube videos in one chat.
- Add timestamp-based references to answers.
- Save vector indexes for faster repeated queries.
- Improve chat history and conversation memory.
- Deploy the application to a cloud platform.

##  Author

**Atharv Chopade**

GitHub: [Atharv-Chopade](https://github.com/Atharv-Chopade)

