// Frontend for the YouTube RAG chatbot.
// Expected backend endpoints:
// POST /api/load-video  { youtube_url, api_key?, model_name? }
// POST /api/chat        { video_id, question }
// POST /api/reset       { video_id } (optional)
const API_BASE = "http://127.0.0.1:8000"; // same origin; change to e.g. "http://127.0.0.1:5000" if needed
const $ = (id) => document.getElementById(id);

const loadForm = $("loadForm");
const loadButton = $("loadButton");
const loadStatus = $("loadStatus");
const youtubeUrl = $("youtubeUrl");
const videoFrame = $("videoFrame");
const videoMeta = $("videoMeta");
const chatMessages = $("chatMessages");
const questionInput = $("questionInput");
const sendButton = $("sendButton");
const chatForm = $("chatForm");
const settingsDialog = $("settingsDialog");

let currentVideoId = null;
let busy = false;

function setStatus(message, type = "") {
  loadStatus.textContent = message;
  loadStatus.className = `status-message ${type}`;
}
function getVideoId(url) {
  try {
    const parsed = new URL(url);
    if (parsed.hostname.includes("youtu.be")) return parsed.pathname.split("/").filter(Boolean)[0] || null;
    if (parsed.hostname.includes("youtube.com")) {
      if (parsed.searchParams.get("v")) return parsed.searchParams.get("v");
      const parts = parsed.pathname.split("/").filter(Boolean);
      if (["embed", "shorts", "live"].includes(parts[0])) return parts[1] || null;
    }
  } catch (_) {}
  return null;
}
function setBusy(value) {
  busy = value;
  loadButton.disabled = value;
  sendButton.disabled = value || !currentVideoId || !questionInput.value.trim();
  questionInput.disabled = !currentVideoId || value;
  $("chatStatus").textContent = value ? "Thinking…" : currentVideoId ? "Video ready for questions" : "Ready when you are";
}
function addMessage(role, content, isTyping = false) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "●" : "✦";
  const bubble = document.createElement("div");
  bubble.className = "message-content";
  if (isTyping) {
    bubble.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
    wrap.id = "typingMessage";
  } else {
    bubble.textContent = content;
  }
  wrap.append(avatar, bubble);
  $("welcomeMessage")?.remove();
  chatMessages.appendChild(wrap);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return wrap;
}
function resetMessages() {
  chatMessages.innerHTML = "";
  const welcome = document.createElement("div");
  welcome.id = "welcomeMessage";
  welcome.className = "welcome-message";
  welcome.innerHTML = `<div class="welcome-sparkle">✦</div><h3>What would you like to know?</h3><p>Ask a question about the loaded video's content.</p><div class="suggestions"><button class="suggestion" data-question="Summarize this video">✧ <span>Summarize this video</span> <b>↗</b></button><button class="suggestion" data-question="What are the main points?">☷ <span>What are the main points?</span> <b>↗</b></button><button class="suggestion" data-question="Explain the key concepts simply">◉ <span>Explain the key concepts simply</span> <b>↗</b></button></div>`;
  chatMessages.appendChild(welcome);
}
async function postJSON(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || data.detail || `Request failed (${response.status})`);
  return data;
}

loadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const url = youtubeUrl.value.trim();
  const guessedId = getVideoId(url);
  if (!guessedId) {
    setStatus("Please enter a valid YouTube video URL.", "error");
    return;
  }
  setBusy(true);
  setStatus("Fetching transcript and building the search index. This can take a little while…");
  try {
    const result = await postJSON("/api/load-video", {
      youtube_url: url,
      api_key: $("apiKey").value.trim(),
      model_name: $("modelName").value.trim()
    });
    currentVideoId = result.video_id || guessedId;
    videoFrame.classList.remove("empty");
    videoFrame.innerHTML = `<iframe src="https://www.youtube-nocookie.com/embed/${encodeURIComponent(currentVideoId)}" title="YouTube video player" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen loading="lazy"></iframe>`;
    videoMeta.innerHTML = `<div class="video-meta-icon">▣</div><div><strong>Video connected</strong><small>${Number(result.chunk_count || 0)} transcript chunks indexed</small></div>`;
    $("engineStatus").textContent = "Video indexed successfully";
    questionInput.placeholder = "Ask anything about this video…";
    questionInput.disabled = false;
    setStatus(`Video indexed successfully${result.chunk_count ? ` · ${result.chunk_count} chunks` : ""}. Ask your first question!`, "success");
    resetMessages();
  } catch (error) {
    setStatus(error.message || "Unable to load this video. Check the backend and try again.", "error");
  } finally {
    setBusy(false);
  }
});

async function askQuestion(question) {
  if (!currentVideoId || busy || !question.trim()) return;
  const cleanQuestion = question.trim();
  addMessage("user", cleanQuestion);
  questionInput.value = "";
  questionInput.style.height = "auto";
  setBusy(true);
  const typing = addMessage("assistant", "", true);
  try {
    const result = await postJSON("/api/chat", { video_id: currentVideoId, question: cleanQuestion });
    typing.remove();
    addMessage("assistant", result.answer || "The backend returned an empty answer.");
  } catch (error) {
    typing.remove();
    addMessage("assistant", `Sorry, I couldn't get an answer: ${error.message}`);
  } finally {
    setBusy(false);
    questionInput.focus();
  }
}
chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  askQuestion(questionInput.value);
});
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 120)}px`;
  sendButton.disabled = busy || !currentVideoId || !questionInput.value.trim();
});
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});
chatMessages.addEventListener("click", (event) => {
  const button = event.target.closest("[data-question]");
  if (button) askQuestion(button.dataset.question);
});
$("clearChat").addEventListener("click", async () => {
  resetMessages();
  if (currentVideoId) {
    try { await postJSON("/api/reset", { video_id: currentVideoId }); } catch (_) {}
  }
});
$("settingsToggle").addEventListener("click", () => settingsDialog.showModal());
$("saveSettings").addEventListener("click", () => settingsDialog.close());
settingsDialog.addEventListener("click", (event) => {
  if (event.target === settingsDialog) settingsDialog.close();
});
setBusy(false);
