import gradio as gr
from inference import extract_products_from_url

def extract(url):
    products = extract_products_from_url(url)
    return products

iface = gr.Interface(
    fn=extract,
    inputs=gr.Textbox(label="Enter URL"),
    outputs="json",
    title="Product Extractor",
)

iface.launch()