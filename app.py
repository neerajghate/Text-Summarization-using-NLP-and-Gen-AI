from flask import Flask, request, jsonify, render_template  # Import render_template
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)

# Load the BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

@app.route('/')
def index():
    return render_template('index.html')  # Ensure your index.html is in a "templates" folder

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize and generate summary
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)
