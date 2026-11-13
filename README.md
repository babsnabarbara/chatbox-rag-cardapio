# Chatbot de Cardápio com RAG

Atendente virtual que lê o cardápio em PDF de um restaurante e responde perguntas sobre ele, indicando de qual página veio a informação.

Feito com Python, LangChain e Gemini, usando RAG (Retrieval-Augmented Generation).

## Motivação

Sou vegana, e sempre que vou a um restaurante preciso investigar o cardápio para descobrir o que posso comer. Pessoas com outras restrições alimentares passam pela mesma coisa. Este chatbot responde perguntas sobre o cardápio usando apenas o que está escrito nele.

## Como funciona

Antes de responder, o sistema busca no documento os trechos mais relevantes para a pergunta e os entrega ao modelo como contexto. O pipeline tem seis etapas:

1. **Carregar:** extrai o texto de cada página do PDF.
2. **Dividir:** corta o texto em pedaços (chunks) de até 1000 caracteres.
3. **Embeddings:** transforma cada chunk em um vetor com o modelo `gemini-embedding-001`.
4. **Guardar:** armazena os vetores no `InMemoryVectorStore`, com cache em disco.
5. **Buscar:** encontra os 3 chunks mais parecidos com a pergunta.
6. **Gerar:** envia a pergunta e os chunks ao `gemini-3.5-flash-lite`, que responde sem inventar informações.

As etapas 1 a 4 rodam uma vez. As etapas 5 e 6 rodam a cada pergunta.

## Estrutura

```
chatbox-rag-cardapio/
├── app.py            # o chatbot
├── modelos.py        # testa o tempo de resposta dos modelos disponíveis
├── cardapio.pdf      # cardápio fictício usado como base
├── requirements.txt  # dependências
└── .gitignore
```

## Como rodar

Clone o repositório e crie um ambiente virtual:

```bash
git clone https://github.com/babsnabarbara/chatbox-rag-cardapio.git
cd chatbox-rag-cardapio
python3 -m venv venv
source venv/bin/activate   # no Windows: venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Gere uma chave gratuita no [Google AI Studio](https://aistudio.google.com) e crie um arquivo `.env` na raiz do projeto:

```
GOOGLE_API_KEY=sua_chave_aqui
```

Execute:

```bash
python3 app.py
```

Na primeira execução, os embeddings são gerados e salvos em `banco_vetores.json`. Nas próximas, são carregados do disco. Se trocar o PDF ou o tamanho dos chunks, apague esse arquivo.

No chat, digite `sair` para encerrar ou `limpar` para limpar a tela.

## Exemplo

```
Qual o horário de funcionamento?
O restaurante funciona de Segunda a Domingo, das 11h às 22h.
     Fontes: 1, 2, 3
```

## Desafios

- **Limite da API:** a primeira versão do cardápio tinha 15 páginas e gerava mais requisições de embedding do que o plano gratuito permite por minuto. Reduzi para 5 páginas e criei um cache em disco para não recalcular os vetores a cada execução.
- **Disponibilidade dos modelos:** alguns modelos estavam descontinuados ou sobrecarregados. O script `modelos.py` testa o tempo de resposta de cada um, e usei o mais rápido.
- **Prompt:** a primeira versão era rígida demais e recusava perguntas razoáveis, como sugerir um prato para o inverno. Ajustei para permitir deduções simples, sem inventar pratos, preços ou horários.

