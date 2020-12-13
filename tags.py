import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords

stop = stopwords.words('english')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    # tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


data = pd.read_json('D:/reviews_Cell_Phones_and_Accessories_5.json/Cell_Phones_and_Accessories_5.json',
                    orient='records', lines=True)
print(data.head())

data['reviewText'] = data['summary'] + ' ' + data['reviewText']
print(data.head())

prods = data['asin'].unique()
print(len(prods))

temp = list(data[data['asin'] == prods[30]]['reviewText'])
print('no. of reviews:', len(temp))
vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=2)
X = vectorizer.fit_transform(temp)
tags = vectorizer.get_feature_names()
cleaned_tags = [i for i in tags if i not in stop]
print(len(cleaned_tags), cleaned_tags)

# step-1
ind = [i for i in range(len(tags)) if tags[i] in cleaned_tags]
ind_sort = sorted(ind, key=lambda t: np.sum(X.toarray(), axis=0)[t], reverse=True)
ultra_tags = [tags[i] for i in ind_sort[:30]]
print(len(ultra_tags), ultra_tags)

# step-2
temp_tags = []
for i in ultra_tags:
    if i.split()[0] in stop:
        temp_tags.append(' '.join(i.split()[1:]))
    else:
        temp_tags.append(i)
new_tags = [i for i in temp_tags if i not in stop]
print(len(new_tags), new_tags)

# step-3
lemmatize_tags = [lemmatize_sentence(i) for i in new_tags]
newer_tags = new_tags
if len(set(lemmatize_tags)) != len(lemmatize_tags):
    newer_tags = []
    new_tags.reverse()
    lemmatize_tags.reverse()
    for i in range(len(lemmatize_tags)):
        flag = 0
        for j in range(i + 1, len(lemmatize_tags)):
            if lemmatize_tags[i] == lemmatize_tags[j]:
                flag = 1
                break
        if flag == 0:
            newer_tags.append(new_tags[i])
    newer_tags.reverse()
print(len(newer_tags), newer_tags)

# step-4
new_tags = []
newer_tags.reverse()
for i in range(len(newer_tags)):
    flag = 0
    synonyms = []
    for syn in wordnet.synsets(newer_tags[i]):
        for l in syn.lemmas():
            synonyms.append(l.name())
    synonyms = list(set([i.replace('_', ' ', ) for i in synonyms]))
    for j in range(i + 1, len(newer_tags)):
        if newer_tags[j] in synonyms:
            flag = 1
            break
    if flag == 0:
        new_tags.append(newer_tags[i])
new_tags.reverse()
print(len(new_tags), new_tags)

# step-5
newer_tags = []
for i in range(len(new_tags)):
    flag = 0
    l = list(range(len(new_tags)))
    l.remove(i)
    for j in l:
        if new_tags[i] in new_tags[j] and len(new_tags[i]) < len(new_tags[j]):
            flag = 1
            break
    if flag == 0:
        newer_tags.append(new_tags[i])
print(len(newer_tags), newer_tags)