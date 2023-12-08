from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)

class TextProcessor:
    def __init__(self):
        # Load SBERT
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def generate_summary(self, input_text, num_sentences=2):
    # Split the input text into sentences
        sentences = [sentence.strip() for sentence in input_text.split('.') if sentence.strip()]

        if not sentences:
            summary = "No valid sentences entered"
            return summary
        
        if num_sentences >= len(sentences):
            summary = f"Error: Number of sentences requested ({num_sentences}) is equal to or exceeds the total number of sentences in the input."
            return summary
        # Generate embeddings for sentences using SBERT
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)

        # Calculate sentence importance scores based on their similarity with the first sentence
        importance_scores = util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings)[0]

        # Sort sentences based on importance scores
        ranked_sentences = sorted(zip(sentences, importance_scores), key=lambda x: x[1], reverse=True)

        # Select the top N sentences as the summary
        summary = '. '.join([sentence[0] for sentence in ranked_sentences[:num_sentences]]) + '.'

        return summary

    
    def analyze_sentiment(self, input_text):
        sentiment_analyzer = SentimentIntensityAnalyzer()

        # Analyze sentiment using nltk
        sentiment_score = sentiment_analyzer.polarity_scores(input_text)
        # Classify sentiment based on polarity score
        if sentiment_score['compound'] >= 0.05:
            return 'Positive'
        elif sentiment_score['compound'] <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

text_processor = TextProcessor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        # Get input text from the form
        input_text = request.form['text']

        # User selected number of sentences in the summary
        num_sentences = int(request.form['num_sentences'])

        # Generate the summary using the TextSummarizer class
        summary = text_processor.generate_summary(input_text, num_sentences)

        # Analyze sentiment
        sentiment = text_processor.analyze_sentiment(input_text)

        return render_template('index.html', input_text=input_text, summary=summary, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
