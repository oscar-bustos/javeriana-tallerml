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

print("Loading M1 (paraphrase-multilingual-MiniLM-L12-v2)...")
m1 = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
e1 = m1.encode(words)

print("Loading M2 (distiluse-base-multilingual-cased-v1)...")
m2 = SentenceTransformer('distiluse-base-multilingual-cased-v1')
e2 = m2.encode(words)

print("Loading M3 (average_word_embeddings_glove.6B.300d)...")
m3 = SentenceTransformer('average_word_embeddings_glove.6B.300d')
e3 = m3.encode(words)

print("Loading M4 (all-MiniLM-L6-v2)...")
m4 = SentenceTransformer('all-MiniLM-L6-v2')
e4 = m4.encode(words)

print("Running PCA...")
pca = PCA(n_components=2)
p1 = pca.fit_transform(e1)
p2 = pca.fit_transform(e2)
p3 = pca.fit_transform(e3)
p4 = pca.fit_transform(e4)

result = []
for i, word in enumerate(words):
    result.append({
        "word": word,
        "x1": round(float(p1[i][0]), 3), "y1": round(float(p1[i][1]), 3),
        "x2": round(float(p2[i][0]), 3), "y2": round(float(p2[i][1]), 3),
        "x3": round(float(p3[i][0]), 3), "y3": round(float(p3[i][1]), 3),
        "x4": round(float(p4[i][0]), 3), "y4": round(float(p4[i][1]), 3)
    })

with open("embeddings_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Successfully saved to embeddings_result.json")
