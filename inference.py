import re
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

model_path = "miisou/ner-product-model" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path).to('cuda' if torch.cuda.is_available() else 'cpu')

# NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="none", device=0 if torch.cuda.is_available() else -1)

def extract_visible_text(url):
    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        [s.decompose() for s in soup(["script", "style", "noscript"])]
        text = soup.get_text(separator='\n', strip=True)
        return text
    except Exception as e:
        return ""


def clean_bert_tokens(tokens):
#    Collecting tokenized words
    word = ""
    for token in tokens:
        if token.startswith("##"):
            word += token[2:]
        else:
            if word:
                word += " "
            word += token
    return word.strip()

def extract_products_from_url(url):
    text = extract_visible_text(url)
    if not text:
        return []

    ner_results = ner_pipeline(text)

    products = []
    current_tokens = []

    for entity in ner_results:
        word = entity["word"]
        label = entity["entity"]


        if word.startswith("##") and label.startswith("B-"):
            label = label.replace("B-", "I-")

        if label.startswith("B-") and not word.startswith("##"):
            if current_tokens:
                products.append(clean_bert_tokens(current_tokens))
            current_tokens = [word]
            
        elif label.startswith("I-") and current_tokens:
            current_tokens.append(word)
            
        else:
            if current_tokens:
                products.append(clean_bert_tokens(current_tokens))
                current_tokens = []

    if current_tokens:
        products.append(clean_bert_tokens(current_tokens))

    clean_products = list(
        set(
            p.strip()
            for p in products
            if len(p.strip()) > 2 and re.search(r"[a-zA-Zа-яА-Я]", p)
        )
    )

    return clean_products
