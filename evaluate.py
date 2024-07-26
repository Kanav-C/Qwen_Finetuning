import time
import psutil
import torch
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_metric

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=model_name)

url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)

db = Qdrant(client=client, embeddings=embeddings, collection_name="hr_db")

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model and move them to the GPU if available
tokenizer = AutoTokenizer.from_pretrained("fine-tuned-qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("fine-tuned-qwen2-0.5B").to(device)

evaluation_data = [
    {"question": "What is the purpose of the referral policy?", "answer": "The purpose of this policy is to recognize and reward employee's contribution in attracting new staff to the organization. Description of the policy The employee referral program at NeoSOFT Technologies is referred to as Refer your Buddy (RUB). The purpose of having this program is to encourage employees at NeoSOFT to play an active part in helping the organization meet its manpower requirements and rewarding them for the same. Process: The employee who has a referral in mind, who he or she thinks meets the required criteria can share the opening with their friend s via Connecto App. Check the ‘Refer a Buddy’ section on Connecto App. The recruitment team will then decide as to whether or not the employee will qualify for Refer your buddy. If the employee does qualify and the candidate is selected then the employee is entitled to the Refer your Buddy reward. Please Note: The employees' role would be limited to the only submission of the resume of the candidate. After this, the regular process of shortlisting and interviews will be applied to the referrals."},
    {"question": "What is the objective of travel policy?", "answer": "The objective of this policy is to provide a set of guidelines which facilitates employee travel & reimbursements for official work lay down band grade-wise entitlement of various travel related expenses ensure all employees have clear and consistent understanding of policies & procedures maximize the company’s ability to negotiate rates, reduce travel exp. and optimized productivity Scope The policies applies to all those traveling on behalf of the Company for company’s business, deputed to attend seminars exhibitions conferences training programs or travelling to client locations Responsibility It will be the responsibility of each Reporting Manager and HOD to ensure that all employee travel meets this objective and reimbursement is made only for actual, reasonable business expenses as per policy. Any expense submitted which does not comply with the guidelines of this policy will not be reimbursed. Non-adherence with policy while on business travel include stringent disciplinary action which may even lead to termination."},
]

bleu_metric = load_metric("bleu", trust_remote_code=True)

def generate_answer(query):
    docs = db.similarity_search_with_score(query=query, k=5)
    context = " ".join([doc.page_content for doc, _ in docs])
    
    input_text = context + " " + query
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=500,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    answer_words = answer.split()
    if len(answer_words) > 200:
        answer = " ".join(answer_words[:200])
    
    return answer

def evaluate_model(evaluation_data):
    references = []
    predictions = []
    
    for data in evaluation_data:
        question = data["question"]
        expected_answer = data["answer"]
        
        generated_answer = generate_answer(question)
        
        references.append([expected_answer.split()])
        predictions.append(generated_answer.split())
    
    results = bleu_metric.compute(predictions=predictions, references=references)
    return results

# Measure performance metrics
start_time = time.time()

# Capture initial CPU and memory usage
initial_cpu_usage = psutil.cpu_percent(interval=None)
initial_memory_usage = psutil.virtual_memory().percent

results = evaluate_model(evaluation_data)

# Measure final CPU and memory usage
final_cpu_usage = psutil.cpu_percent(interval=None)
final_memory_usage = psutil.virtual_memory().percent

end_time = time.time()
total_time = end_time - start_time
average_time_per_query = total_time / len(evaluation_data)

# Calculate model size in MB and GB
model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)  # in MB
model_size_gb = model_size_mb / 1024  # in GB

print(f"BLEU score: {results['bleu']}")
print(f"Total response time: {total_time:.2f} seconds")
print(f"Average response time: {average_time_per_query:.2f} seconds")
print(f"Model size: {model_size_mb:.2f} MB ({model_size_gb:.2f} GB)")
print(f"CPU usage before evaluation: {initial_cpu_usage}%")
print(f"CPU usage after evaluation: {final_cpu_usage}%")
print(f"Memory usage before evaluation: {initial_memory_usage}%")
print(f"Memory usage after evaluation: {final_memory_usage}%")
print(f"GPU used: {torch.cuda.is_available()}")
