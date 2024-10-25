import transformers
import torch
from torch import cuda, bfloat16
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

import streamlit as st
from streamlit_chat import message
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.chat_models import ChatOpenAI
#from langchain.chains import ConversationalRetrievalChain
#from langchain.document_loaders.csv_loader import CSVLoader
#from langchain.vectorstores import FAISS
#import tempfile

# Definição do modelo
model_id = '../../llama2-virtual-sales-assistant'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# Configuração da quantização para carregar o modelo usando menor quantidade de memória na GPU
# Requer biblioteca bitsandbytes
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# Carregamento do modelo do HuggingFace
model_config = transformers.AutoConfig.from_pretrained(model_id)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto'
)

# Habilitação do modo de avaliação do modelo
model.eval()
print(f"Modelo carregado em {device}")

# Carregamento do tokenizer do modelo
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

# Definição dos critérios de parada
stop_list = ['\nHuman:', '\n```\n']
stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]

# Transformação dos tokens de parada em tensor do PyTorch
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

# Criação de função que verifica a existência de critério de parada
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False
stopping_criteria = StoppingCriteriaList([StopOnTokens()])

# Definição de parâmetros adicionais do modelo
generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

# Implementação do pipeline do HuggingFace no Langchain
llm = HuggingFacePipeline(pipeline=generate_text)

# Carregamento dos dados
loader_1 = CSVLoader(file_path="../dataset/olist_order_items_dataset.csv",
        encoding='utf-8',
        source_column='order_id',
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["order_id","order_item_id","product_id","seller_id","shipping_limit_date","price","freight_value"]
    },
)
data_1 = loader_1.load()
loader_2 = CSVLoader(file_path="../dataset/olist_orders_dataset.csv",
        encoding='utf-8',
        source_column="customer_id",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["order_id","customer_id","order_status","order_purchase_timestamp","order_approved_at","order_delivered_carrier_date","order_delivered_customer_date","order_estimated_delivery_date"]
    },
)
data_2 = loader_2.load()
loader_3 = CSVLoader(file_path="../dataset/olist_products_dataset.csv",
        encoding='utf-8',
        source_column='product_id',
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ['product_id','id_Flipart','actual_price','Usd-price','average_rating','brand','category','crawled_at','description','discount','images','out_of_stock','pid','product_details','seller','selling_price','sub_category','title','url']
    },
)
data_3 = loader_3.load()
data = data_1 + data_2 + data_3

# Carregamento de modelo e criação dos embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Armazenamento dos embeddings
vectorstore = FAISS.from_documents(data, embeddings)

# Inicialização da chain
chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

def conversational_chat(query):
        
    result = chain({"question": query, 
    "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
        
    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! How can I help you?"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey!"]
        
#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
            
        user_input = st.text_input("Query:", placeholder="Type here", key='input')
        submit_button = st.form_submit_button(label='Send')
            
    if submit_button and user_input:
        output = conversational_chat(user_input)
            
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")