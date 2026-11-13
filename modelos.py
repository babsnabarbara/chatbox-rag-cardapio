import time
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

# 1. Lista os modelos "flash" que sabem escrever texto
candidatos = []
for m in client.models.list():
    nome = m.name
    if "generateContent" in m.supported_actions and "flash" in nome:
        if not any(x in nome for x in ["image", "tts", "audio", "live"]):
            candidatos.append(nome)

print(f"Testando {len(candidatos)} modelos...\n")

# 2. Manda uma pergunta minúscula para cada um e mede o tempo
for nome in candidatos:
    inicio = time.perf_counter()
    try:
        client.models.generate_content(model=nome, contents="Responda só: ok")
        print(f"✅ {nome}: {time.perf_counter() - inicio:.1f}s")
    except Exception as erro:
        print(f"❌ {nome}: {str(erro)[:70]}")