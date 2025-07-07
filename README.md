# NER_product_extraction_from_furniture_stores

## Description

Main goal of this project was to build a NER model, which takes URL as an input and gives list of extracted products from website (particularly furniture). Model did its job below average achieving 40% recall and 44$ precision, due to disability to make a hand-made labeling and quality of some of the websites. Also, the evaluation dataset consisted of 10% of links to enlarge training dataset and it can affect metrics.

This project involves:

+ parsing websites
+ automatic labeling using filters by keywords and patterns
+ BIO-tokenization and preparing dataset
+ fine-tuning bert-base-cased for NER
+ analyzing and processing result for clear output
+ deploying on huggingface

## Project structure

```
├── LICENSE
├── README.md             ← readme
├── URL_list.csv          ← initial URL list 
├── df_web_texts_for_ner.csv ← csv with already parsed text for working links to save some time while experimenting.
├── app.py                ← gradio app
├── inference.py          ← inference for gradio app with ner pipeline and results processing
|── notebook.ipynb        ← notebook with parsing, labeling, tokenizing and training model. Somewhat dirty because it was made to show online
├── requirements.txt      ← requirements (made by pip freeze)
```

## Setup

1. Clone repository

```
git clone https://github.com/miisou/NER_product_extraction_from_furniture_stores.git
cd NER_product_extraction_from_furniture_stores
```

2. Install requirements

```
pip install transformers torch beautifulsoup4 requests numpy pandas datasets tqdm gradio
```

## Running the application

This model is publicly available at https://huggingface.co/spaces/miisou/NER_product_extraction_from_furniture_stores

You can still run it locally after running notebook.ipynb, inference.py and app.py in this order. 

