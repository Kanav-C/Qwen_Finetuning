from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import fitz
import os

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=model_name)

url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)
db = Qdrant(client=client, embeddings=embeddings, collection_name="ft_db")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

def prepare_text_data_from_pdfs(pdf_folder):
    texts = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            texts.append(text)
    return texts

pdf_folder = "C:\\Users\\DELL\\Downloads\\HR\\HR"
texts = prepare_text_data_from_pdfs(pdf_folder)
dataset = Dataset.from_dict({"text": texts})

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

def add_labels(examples):
    examples["labels"] = examples["input_ids"].copy()
    return examples

tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

trainer.train()

model.save_pretrained("fine-tuned-qwen2-0.5B")
tokenizer.save_pretrained("fine-tuned-qwen2-0.5B")

def generate_answer(query):
    docs = db.similarity_search_with_score(query=query, k=5)
    context = " ".join([doc.page_content for doc, _ in docs])
    
    input_text = context + " " + query
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    
    outputs = model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    answer = " ".join(answer.split()[:100])
    
    return answer

query = "What are the company's data privacy policies?"
answer = generate_answer(query)
print(f"Query: {query}\nAnswer: {answer}")
