from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

class TextSummarizer:
    def __init__(self):
        # Load the SBERT model
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def generate_summary(self, input_text, num_sentences=2):
        # Split the input text into sentences
        sentences = input_text.split('.')

        # Generate embeddings for sentences using SBERT
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)

        # Calculate sentence importance scores based on their similarity with the first sentence
        importance_scores = util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings)[0]

        # Sort sentences based on importance scores
        ranked_sentences = sorted(zip(sentences, importance_scores), key=lambda x: x[1], reverse=True)

        # Select the top N sentences as the summary
        summary = '. '.join([sentence[0] for sentence in ranked_sentences[:num_sentences]]) + '.'
        
        return summary

text_summarizer = TextSummarizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        # Get input text from the form
        input_text = request.form['text']

        # User-selected number of sentences in the summary
        num_sentences = int(request.form['num_sentences'])

        # Generate the summary using the TextSummarizer class
        summary = text_summarizer.generate_summary(input_text, num_sentences)

        return render_template('index.html', input_text=input_text, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
