from sentence_transformers import SentenceTransformer
import faiss
import json

# model=SentenceTransformer('all-MiniLM-L6-v2')
model=SentenceTransformer('bert-base-nli-mean-tokens')
# model=SentenceTransformer('paraphrase-MiniLM-L6-v2')

sentences=[]

with open('data.json', 'r') as f:
    data=f.read()
    data=json.loads(data)
    for key in data:
        for ke in data[key]:
            des=""
            for k in data[key][ke]:
                des+=f"{k} : {data[key][ke][k]}\n"
            
            sentences.append(des)
text_embeddings = model.encode(sentences, convert_to_tensor=True)

d=text_embeddings.shape[1]
print(text_embeddings.shape)
index=faiss.IndexFlatL2(d)

index.add(text_embeddings)

# query ="women is lie in people's heart"
query='wedding clothes'

query_embedding = model.encode(query, convert_to_tensor=True)

query_embedding = query_embedding.reshape(1, -1)

k=3

distances, indices = index.search(query_embedding, k)

print(f"Query: {query}")
print(f"\nTop {k} Matches:")
for i, idx in enumerate(indices[0]):
    print(f"---------- Product {i+1} : {idx} ---------- \n{sentences[idx]} (Distance: {distances[0][i]:.4f}) \n ---------------------------------------")
