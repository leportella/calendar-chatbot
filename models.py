import json
import random

import numpy as np
import spacy


nlp = spacy.load('pt')


def get_list_of_words(intents):
    words = []
    classes = []
    documents = []

    for intent in intents['classes']:
        for pattern in intent['patterns']:
            doc = nlp(pattern)
            w = [token.lemma_.lower() for token in doc if not token.is_punct]
            words.extend(w)

            documents.append((w, intent['intent']))

            if intent['intent'] not in classes:
                classes.append(intent['intent'])

    words = sorted(list(set(words)))
    return classes, documents, words


def get_x_y_from_words(classes, documents, words):
    output = []
    output_empty = [0] * len(classes)

    x = []
    y = []

    for doc in documents:
        features = []
        pattern_words = doc[0]
        pattern_words = [word.lower() for word in pattern_words]
        for w in words:
            features.append(1) if w in pattern_words else features.append(0)

        labels = list(output_empty)
        labels[classes.index(doc[1])] = 1

        x.append(features)
        y.append(labels)
        return x, y


with open('intents.json') as json_data:
    intents = json.load(json_data)

classes, documents, words = get_list_of_words(intents)
x, y = get_x_y_from_words(classes, documents, words)
