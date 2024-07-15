from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load the model and tokenizer from local files
model_path = "./local_model"
tokenizer_path = "./local_tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Define label names
# Note: For the SST-2 dataset, there are only two labels
label_names = ['negative', 'positive']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        verse = request.form['verse']
        sentiment = analyze_sentiment(verse)
        return render_template('index.html', result=sentiment)
    return render_template('index.html', result=None)

def analyze_sentiment(verse):
    inputs = tokenizer(verse, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    confidence = probabilities[0][predicted_class].item()
    sentiment = label_names[predicted_class]
    
    return {
        "verse": verse,
        "sentiment": sentiment,
        "confidence": f"{confidence:.2%}"
    }

if __name__ == '__main__':
    app.run(debug=True)