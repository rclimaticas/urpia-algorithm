import numpy as np
import requests
from scipy.spatial.distance import cdist
from flask import Flask

app = Flask(__name__)

# URLs configuráveis
users_url = "https://psychic-parakeet-q5grgjr9vwg39gw9-3333.app.github.dev/profile"
impacts_url = "https://psychic-parakeet-q5grgjr9vwg39gw9-3333.app.github.dev/impacts"

# Mapeamento dos biomas
biomes = [
    "Mata Atlântica", "Caatinga", "Amazônia", "Pampas", "Pantanal", "Cerrado", "Zonas Urbanas"
]

# Mapeamento dos povos
peoples = [
    "Agricultor Familiar", "Indígenas", "Quilombolas", "Fundo de Pasto", "Gerais",
    "Pescadores Ribeirinhos", "Pescadores/Marisqueiras", "Cidades", "Geraizeiros",
    "Religiosos", "Ciganos", "Nômades", "Outros"
]

# Funções auxiliares
def get_users():
    response = requests.get(users_url)
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list):
            return data
        else:
            print({"error": "A estrutura de dados não é uma lista como esperado."})
            return None
    else:
        print({"error": "Não foi possível acessar os dados dos usuários."})
        return None

def get_impact():
    response = requests.get(impacts_url)
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        else:
            print({"error": "A estrutura de dados de impactos não é uma lista ou está vazia."})
            return None
    else:
        print({"error": "Não foi possível acessar os dados dos impactos."})
        return None

def impact_to_vector(impact_data):
    impact_vector = [0] * (len(biomes) + len(peoples))
    for i, biome in enumerate(biomes):
        if biome in impact_data.get("biomes", []):
            impact_vector[i] = 1
    for i, people in enumerate(peoples):
        if people in impact_data.get("affectedCommunity", []):
            impact_vector[len(biomes) + i] = 1
    return np.array(impact_vector)

def calculate_distances(user_matrix, impact_vector):
    return cdist(user_matrix, impact_vector.reshape(1, -1), metric='euclidean').flatten()

def process_last_impact():
    impact_data = get_impact()
    if not impact_data:
        return {"error": "Não foi possível processar o impacto."}

    users_data = get_users()
    if not users_data:
        return {"error": "Não foi possível acessar os usuários."}

    user_matrix = []
    for item in users_data:
        user_row = [0] * (len(biomes) + len(peoples))
        for i, biome in enumerate(biomes):
            if biome in item.get("themesBiomes", []):
                user_row[i] = 1
        for i, people in enumerate(peoples):
            if people in item.get("themesCommunities", []):
                user_row[len(biomes) + i] = 1
        user_matrix.append({
            "id": item.get("id", None),
            "email": item.get("email", None),
            "membership": np.array(user_row)
        })

    user_matrix_np = np.array([user['membership'] for user in user_matrix])
    impact_vector = impact_to_vector(impact_data)
    distances = calculate_distances(user_matrix_np, impact_vector)

    K = 3
    nearest_neighbors_indices = np.argsort(distances)[:K]

    results = []
    for idx in nearest_neighbors_indices:
        user_id = user_matrix[idx]["id"]
        user_email = user_matrix[idx]["email"]
        user_distance = distances[idx]
        results.append({
            "id": user_id,
            "email": user_email,
            "distance": round(user_distance, 4)
        })

    return {"impact_id": impact_data["id"], "nearest_neighbors": results}

# Rota principal para processar o impacto
@app.route("/")
def index():
    result = process_last_impact()
    return result, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
