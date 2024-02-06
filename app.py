from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv()
import weaviate
import os
#os.environ["OPENAI_API_KEY"] = getpass.getpass("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")
WEAVIATE_URL = "http://my-sandbox-cluster-fp8df6e0.weaviate.network"
#os.environ["WEAVIATE_API_KEY"] = getpass.getpass("WEAVIATE_API_KEY:")
WEAVIATE_API_KEY = "xnIEYLWengRF9JL3kRGmT5X7L8fjG8IMgR0z"

# client = weaviate.Client(
#     url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY)
# )
client = weaviate.connect_to_wcs(
    cluster_url="http://my-sandbox-cluster-fp8df6e0.weaviate.network",  # Replace with your WCS URL
    auth_credentials=weaviate.auth.AuthApiKey("xnIEYLWengRF9JL3kRGmT5X7L8fjG8IMgR0z")  # Replace with your WCS key
)

loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = Weaviate.from_documents(docs, embeddings, client=client,weaviate_url=WEAVIATE_URL, by_text=False)

query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(docs[0].page_content)

