import os
from flask import Flask, request, render_template
from flask_cors import CORS
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import openai

# =====================================================
# 🔑 إعداد مفتاح OpenAI من متغير البيئة
# =====================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# =====================================================
# 📂 الملفات
# =====================================================
# تأكدي أن هذه الملفات موجودة داخل الريبو على GitHub
FILE_1 = "data/AI_Broadcast_Log.xlsx"
FILE_2 = "data/broadcast_ai_knowledge_Bilingual.xlsx"
TEMPLATE_PATH = "templates"  # يجب وضع chat.html هنا

# =====================================================
# 🧠 ردود التحية العامة
# =====================================================
def is_general_question(text):
    greetings = [
        "مرحبا", "أهلا", "اهلا", "السلام عليكم",
        "hi", "hello", "hey", "good morning", "good evening"
    ]
    return any(g.lower() in text.lower() for g in greetings)

def general_reply(text):
    if "السلام عليكم" in text:
        return "وعليكم السلام ورحمة الله وبركاته 🌸"
    if any(w in text.lower() for w in ["مرحبا", "اهلا", "hello", "hi"]):
        return "أهلاً وسهلاً! كيف أقدر أساعدك؟ 😊"
    return "مرحباً! كيف يمكنني مساعدتك؟"

# =====================================================
# 📄 تحميل مستندات Excel
# =====================================================
def load_documents():
    docs = []
    for file in [FILE_1, FILE_2]:
        df = pd.read_excel(file)
        for _, row in df.iterrows():
            docs.append(" ".join(row.astype(str)))
    return docs

# =====================================================
# 🔎 بناء قاعدة المتجهات FAISS
# =====================================================
print("🔄 Loading documents and building embeddings...")
documents = load_documents()
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(documents, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"✅ {len(documents)} documents indexed.")

# =====================================================
# 🤖 دالة التفاعل مع GPT
# =====================================================
def chat_with_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response['choices'][0]['message']['content']

# =====================================================
# 🔎 البحث والرد
# =====================================================
def answer_question(question, top_k=3):
    q_emb = model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    context = "\n".join([documents[i] for i in I[0]])
    prompt = f"استخدم هذه المعلومات للإجابة على السؤال:\n{context}\nالسؤال: {question}"
    return chat_with_openai(prompt)

# =====================================================
# 🌐 Flask App
# =====================================================
app = Flask(__name__, template_folder=TEMPLATE_PATH)
CORS(app)

history = []

@app.route("/", methods=["GET", "POST"])
def chat_page():
    global history

    if request.method == "POST":
        user_message = request.form.get("message", "").strip()

        if user_message:
            history.append({"role": "user", "content": user_message})

            if is_general_question(user_message):
                bot_reply = general_reply(user_message)
            else:
                bot_reply = answer_question(user_message)

            history.append({"role": "bot", "content": bot_reply})

    return render_template("chat.html", history=history)

# =====================================================
# 🚀 تشغيل Flask على Render
# =====================================================
if __name__ == "__main__":
    # Render يعطي المنفذ من متغير البيئة PORT
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
