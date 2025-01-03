import os
import json
from flask import Flask, request, abort
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent

# Initialize Flask app
app = Flask(__name__)



configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
line_handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET")) # overlap with the python handler when deploying with versel so change it


# Load FAQ dataset
FAQ_DATASET_PATH = "faq_dataset.json"
try:
    with open(FAQ_DATASET_PATH, "r", encoding="utf-8") as f:
        faq_data = json.load(f)
except FileNotFoundError:
    print(f"FAQ file not found at {FAQ_DATASET_PATH}.")
    faq_data = []

faq_questions = [item["question"] for item in faq_data]
faq_answers = [item["answer"] for item in faq_data]

# Precompute FAQ embeddings using TF-IDF
def tokenize_chinese(texts):
    return [" ".join(jieba.cut(text)) for text in texts]

try:
    CHINESE_STOP_WORDS = "baidu_stopwords.txt"
    with open(CHINESE_STOP_WORDS, "r", encoding="utf-8") as f:
        zh_stop_words = f.read().splitlines()

    ENGLISH_STOP_WORDS = "EN-Stopwords.txt"
    with open(ENGLISH_STOP_WORDS, "r", encoding="utf-8") as f:
        en_stop_words = f.read().splitlines()

    stop_words = zh_stop_words + en_stop_words
except FileNotFoundError as e:
    print(f"Stop words file not found: {e}")
    stop_words = []

faq_questions_tokenized = tokenize_chinese(faq_questions)
vectorizer = TfidfVectorizer(stop_words=stop_words).fit(faq_questions_tokenized)
faq_embeddings = vectorizer.transform(faq_questions_tokenized)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        line_handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.error("Invalid signature. Check your channel access token/channel secret.")
        abort(400)

    return 'OK'

@line_handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_message = event.message.text
    response = find_best_response(user_message)

    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        try:
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=response)]
                )
            )
        except Exception as e:
            app.logger.error(f"Error sending reply: {e}")

def find_best_response(user_message):
    try:
        user_message_tokenized = tokenize_chinese([user_message])
        user_embedding = vectorizer.transform(user_message_tokenized)

        similarities = cosine_similarity(user_embedding, faq_embeddings)
        best_match_index = similarities.argmax()
        best_match_score = similarities[0, best_match_index]
        app.logger.info(f"Best match score: {best_match_score}")

        THRESHOLD = 0.5
        if best_match_score >= THRESHOLD:
            return faq_answers[best_match_index]
        else:
            return "抱歉，目前無法處理您的請求，請稍後再試。"
    except Exception as e:
        app.logger.error(f"Error finding best response: {e}")
        return "抱歉，目前無法處理您的請求，請稍後再試。"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port)
