import tkinter as tk
from tkinter import ttk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag, word_tokenize
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
from wordcloud import WordCloud
from itertools import count
from matplotlib.animation import FuncAnimation

# Download necessary NLTK resources (run this once)
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Function to generate and display word cloud
def generate_wordcloud(text):
    text = text.lower()
    text = "".join([char for char in text if char.isalnum() or char == " "])

    stop_words = set(stopwords.words('english'))
    stop_words.update(["the", "a", "is", "of"])
    wordcloud = WordCloud(width=600, height=400, stopwords=stop_words, background_color='black').generate(text)

    plt.figure(figsize=(8, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Function to perform sentiment analysis using TextBlob
def perform_sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment

    if sentiment.polarity > 0:
        return "Positive"
    elif sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to analyze text and predict personality
def analyze_text():
    text = text_entry.get()
    if text:
        try:
            vader_analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download('vader_lexicon')
            vader_analyzer = SentimentIntensityAnalyzer()

        sentiment_scores = vader_analyzer.polarity_scores(text)
        pos_tags = pos_tag(word_tokenize(text))

        # Analyzing Big Five personality traits
        personality_traits = []

        if sentiment_scores['compound'] > 0.5:
            personality_traits.append("Optimistic")
        elif sentiment_scores['compound'] < -0.5:
            personality_traits.append("Critical")
        else:
            personality_traits.append("Neutral")

        outgoing_adverbs = sum(1 for word, tag in pos_tags if tag.startswith('RB'))
        if outgoing_adverbs > len(pos_tags) // 5:
            personality_traits.append("Outgoing")

        generate_wordcloud(text)

        # Descriptions of predicted personality traits
        personality_description = ""
        if "Optimistic" in personality_traits:
            personality_description += "You appear to have an optimistic outlook."
        elif "Critical" in personality_traits:
            personality_description += "Your analysis tends to be critical and skeptical."
        elif "Neutral" in personality_traits:
            personality_description += "You seem to have a neutral stance."

        if "Outgoing" in personality_traits:
            personality_description += " Additionally, you exhibit outgoing characteristics."

        # Update labels on the first page with personality traits and description
        personality_label.config(text=f"Predicted Personality Traits: {', '.join(personality_traits)}")
        description_label.config(text=personality_description)
        accuracy_label.config(text="Accuracy: High")  # Placeholder for accuracy estimation

        # Calculate Big Five personality traits percentages
        traits_count = {'Openness': 0, 'Conscientiousness': 0, 'Extraversion': 0, 'Agreeableness': 0, 'Neuroticism': 0}

        for word, tag in pos_tags:
            if tag.startswith('NN'):
                traits_count['Openness'] += 1
            elif tag.startswith('VB') or tag.startswith('MD'):
                traits_count['Conscientiousness'] += 1
            elif tag.startswith('JJ') or tag.startswith('RB'):
                traits_count['Extraversion'] += 1
            elif tag.startswith('PRP') or tag.startswith('DT') or tag.startswith('IN'):
                traits_count['Agreeableness'] += 1
            elif tag.startswith('PRP$') or tag.startswith('NN') or tag.startswith('CD'):
                traits_count['Neuroticism'] += 1

        total_words = sum(traits_count.values())
        percentages = [count / total_words * 100 for count in traits_count.values()]

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(traits_count.keys(), [0] * len(traits_count), color=['blue', 'orange', 'green', 'red', 'purple'])

        def animate(i):
            for bar, percentage in zip(bars, percentages):
                bar.set_height(min(bar.get_height() + 1, percentage))

        anim = FuncAnimation(fig, animate, frames=range(100), interval=50)
        plt.xlabel('Big Five Personality Traits')
        plt.ylabel('Percentage')
        plt.title('Big Five Personality Traits Analysis BY CU STUDENTS ')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.show()

# Create the main window
window = tk.Tk()
window.title("Personality Prediction Analyzer - MANAS KATURI ")
window.configure(bg="#202124")

# Add UI elements with dark theme
description_label = tk.Label(window, text="Welcome to the Personality Prediction Analyzer!", bg="#202124", fg="white", font=("Arial", 18, "bold"))
description_label.pack(pady=20)

tool_info_label = tk.Label(window, text="This tool predicts your personality based on the entered text.\nDeveloped by Manas Prakash Katuri ", bg="#202124", fg="white", font=("Arial", 14))
tool_info_label.pack(pady=10)

text_label = tk.Label(window, text="Enter Text to Analyze:", bg="#202124", fg="white", font=("Arial", 14))
text_label.pack(pady=10)

text_entry = tk.Entry(window, width=50, font=("Arial", 12))
text_entry.pack(pady=5)

analyze_button = tk.Button(window, text="Analyze Text", command=analyze_text, bg="#3399ff", fg="white", font=("Arial", 12))
analyze_button.pack(pady=5)

personality_label = tk.Label(window, text="", bg="#202124", fg="white", font=("Arial", 14))
personality_label.pack(pady=10)

description_label = tk.Label(window, text="", bg="#202124", fg="white", font=("Arial", 12), wraplength=400, justify="center")
description_label.pack(pady=5)

accuracy_label = tk.Label(window, text="", bg="#202124", fg="white", font=("Arial", 14))
accuracy_label.pack(pady=5)

window.mainloop()
