from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer


def fileReader():
    # file handling
    print("Enter your file name below. ")
    file_name = input("Enter: ")
    try:
        with open(file_name, 'r') as file:
            lines = file.read()
            return lines
    except FileNotFoundError:
        print(f"{file_name} does not exists")


def textCleaning():
    lines = fileReader()
    # stop words
    stop_words = set(stopwords.words('english'))
    # Tokenization
    tokens = word_tokenize(lines)
    filtered_sentence = []
    for word in tokens:
        if word not in stop_words:
            filtered_sentence.append(word.lower())
    # Lemmatizing the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_sentence]
    processed_txt = ' '.join(lemmatized_tokens)
    return processed_txt


def getSentiments():
    # perform analysis
    processed_txt = textCleaning()
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(processed_txt)
    return scores


# printing results
def getResults():
    scores = getSentiments()
    for k, v in scores.items():
        print(f"{k}: {v}")


getResults()
