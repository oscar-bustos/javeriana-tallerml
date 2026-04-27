import json
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

words = [
    "delantero", "portero", "defensa", "mediocampista", "árbitro", "entrenador", "suplente", "capitán", "messi", "ronaldo",
    "chutar", "pasar", "regatear", "atajar", "cabecear", "correr", "saltar", "defender", "atacar", "marcar",
    "balón", "portería", "red", "guantes", "guayos", "trofeo", "estadio", "cancha",
    "gol", "falta", "penal", "tiro libre", "córner", "tarjeta roja", "tarjeta amarilla", "var",
    "baloncesto", "tenis", "raqueta", "nadar", "piscina", "auto", "conducir"
]

print("Loading Model 1 (paraphrase-multilingual-MiniLM-L12-v2)...")
model1 = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
emb1 = model1.encode(words)

print("Loading Model 2 (distiluse-base-multilingual-cased-v1)...")
model2 = SentenceTransformer('distiluse-base-multilingual-cased-v1')
emb2 = model2.encode(words)

print("Running PCA...")
pca = PCA(n_components=2)
proj1 = pca.fit_transform(emb1)
proj2 = pca.fit_transform(emb2)

result = []
for i, word in enumerate(words):
    result.append({
        "word": word,
        "x1": round(float(proj1[i][0]), 3),
        "y1": round(float(proj1[i][1]), 3),
        "x2": round(float(proj2[i][0]), 3),
        "y2": round(float(proj2[i][1]), 3)
    })

with open("embeddings_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Successfully saved to embeddings_result.json")
