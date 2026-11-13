from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import subprocess
#Primeira fase do RAG: carregar os documentos e extrair o texto
load_dotenv()

reader = PdfReader("cardapio.pdf")
docs = []

for num, pag in enumerate(reader.pages, start=1):
    text = pag.extract_text()
    docs.append(Document(page_content=text, metadata={"page": num}))

#print(f"Páginas carregadas: {len(docs)}")
#print(docs[0].page_content[:300])

#Segunda fase do RAG: Split - cortar o texto em chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)


#Terceira fase do RAG: Embeddings - transformando os chunks em vetores
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


#Quarta fase do RAG: Guardar - convertendo cada chunk em vetor (usando a fase 3)
#e armazenando o vetor com texto e metadados no vectorstore, que é um banco de dados em memória
ARQUIVO_BANCO = "banco_vetores.json"
if os.path.exists(ARQUIVO_BANCO):
    #print(f"Carregando vetores do disco")
    vectorstore = InMemoryVectorStore.load(ARQUIVO_BANCO, embedding=embeddings)
else:
    #print(f"Gerando vetores pela primeira vez e salvando em disco")
    vectorstore = InMemoryVectorStore.from_documents(chunks, embedding=embeddings)
    vectorstore.dump(ARQUIVO_BANCO)



llm = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', timeout=60,)
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Você é o atendente virtual do restaurante vegano Verde Sabor. "
     "Responda usando APENAS as informações do contexto abaixo. "
     "Se a resposta não estiver no contexto, diga que não encontrou essa informação no cardápio. "
     "Nunca invente pratos, preços ou horários. "
     "Se a pergunta exigir comparar o cardápio inteiro (como 'o prato mais barato'), "
     "responda em texto simples, sem formatação, nem emojis"
     "Contexto:\n{context}"),
    ("human", "{question}"),
])

chain = prompt | llm #a saída do prompt entra na llm
def format_context(docs):
    return "\n\n---\n\n".join(f"[Página {d.metadata['page']}]\n{d.page_content}" for d in docs)

while True:
    question = input("Digite sua pergunta, ou sair para encerrar\n")
    if question.lower() == "sair":
        break

    #Quinta fase do RAG: Busca - encontrar os vetores mais semelhantes ao texto da pergunta
    results = vectorstore.similarity_search(question, k=3)

    #Sexta fase do RAG: Gerar a resposta
    context = format_context(results)
    response = chain.invoke({"context": context, "question": question})
    pages = sorted({d.metadata["page"] for d in results})
    print(response.text)
    print(f"     Fontes: {', '.join(map(str,pages))}\n")
