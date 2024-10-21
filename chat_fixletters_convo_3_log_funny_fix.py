from components.assets import arrow_circle_icon, github_icon
from components.chat import chat, chat_form, chat_message, chat_messages
import asyncio
import modal
from fasthtml.common import *
import fastapi
import logging
from transformers import AutoTokenizer

# Constants
MODELS_DIR = "/llama_mini"
MODEL_NAME = "Llama-3.2-3B"  # Adjusted MODEL_NAME
FAISS_DATA_DIR = "/faiss_data"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
USERNAME = "c123ian"
APP_NAME = "rag-chatbot"

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize the tokenizer
# Replace 'gpt2' with a tokenizer compatible with your model if available
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Download the model weights
try:
    volume = modal.Volume.lookup("llama_mini", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with the appropriate script")

# Define the Modal image with required dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "vllm==0.5.3post1",
    "python-fasthtml==0.4.3",
    "aiohttp",          # For asynchronous HTTP requests
    "faiss-cpu",        # For FAISS
    "sentence-transformers",
    "pandas",
    "numpy",
    "huggingface_hub",
    "transformers"      # Added for the tokenizer
)

# Define the FAISS volume
try:
    faiss_volume = modal.Volume.lookup("faiss_data", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Create the FAISS data volume first with download_faiss.py")

# Define the Modal app
app = modal.App(APP_NAME)

# vLLM server implementation with model path handling
@app.function(
    image=image,
    # gpu=modal.gpu.L4(count=1),
    gpu=modal.gpu.A100(count=1, size="40GB"),
    container_idle_timeout=10 * 60,
    timeout=24 * 60 * 60,
    allow_concurrent_inputs=100,
    volumes={MODELS_DIR: volume},
)
@modal.asgi_app()
def serve_vllm():
    import os
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.logger import RequestLogger
    import fastapi
    from fastapi.responses import StreamingResponse, JSONResponse
    import uuid
    import asyncio
    from typing import Optional, AsyncGenerator

    # Function to find the model path by searching for 'config.json'
    def find_model_path(base_dir):
        for root, dirs, files in os.walk(base_dir):
            if "config.json" in files:
                return root
        return None

    # Check if model files exist
    model_path = find_model_path(MODELS_DIR)
    if not model_path:
        raise Exception(f"Could not find model files in {MODELS_DIR}")

    print(f"Initializing AsyncLLMEngine with model path: {model_path}")

    # Create a FastAPI app
    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com",
        version="0.0.1",
        docs_url="/docs",
    )

    # Create an `AsyncLLMEngine`, the core of the vLLM server.
    engine_args = AsyncEngineArgs(
        model=model_path,  # Use the found model path
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Get model config using the robust event loop handling
    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    # Initialize OpenAIServingChat
    request_logger = RequestLogger(max_log_len=256)  # Adjust max_log_len as needed
    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        [MODEL_NAME],  # served_model_names
        "assistant",   # response_role
        lora_modules=None,  # Adjust if you're using LoRA
        prompt_adapters=None,  # Adjust if you're using prompt adapters
        request_logger=request_logger,
        chat_template=None,  # Adjust if you have a specific chat template
    )

    @web_app.get("/v1/completions")
    async def get_completions(prompt: str, max_tokens: int = 6000, stream: bool = False):
        request_id = str(uuid.uuid4())
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.5,  # Encourage the model to use new tokens
            stop=["User:", "\n\n"]
        )

        async def completion_generator() -> AsyncGenerator[str, None]:
            try:
                full_response = ""
                assistant_prefix_removed = False
                last_yielded_position = 0
                async for result in engine.generate(prompt, sampling_params, request_id):
                    if len(result.outputs) > 0:
                        new_text = result.outputs[0].text
                        if not assistant_prefix_removed:
                            new_text = new_text.split("Assistant:")[-1].lstrip()
                            assistant_prefix_removed = True

                        if len(new_text) > last_yielded_position:
                            new_part = new_text[last_yielded_position:]
                            yield new_part
                            last_yielded_position = len(new_text)

                        full_response = new_text

                        # Removed premature stopping condition
                        # if full_response.strip().endswith((".", "!", "?")):
                        #     break
            except Exception as e:
                yield str(e)

        if stream:
            return StreamingResponse(
                completion_generator(), media_type="text/event-stream"
            )
        else:
            completion = ""
            async for chunk in completion_generator():
                completion += chunk
            return JSONResponse(content={"choices": [{"text": completion.strip()}]})

    return web_app

# FastHTML web interface implementation with RAG
@app.function(
    image=image,
    volumes={FAISS_DATA_DIR: faiss_volume},
)
@modal.asgi_app()
def serve_fasthtml():
    import faiss
    import os

    # Paths to FAISS index and documents
    FAISS_INDEX_PATH = os.path.join(FAISS_DATA_DIR, "faiss_index.bin")
    DATA_PICKLE_PATH = os.path.join(FAISS_DATA_DIR, "data.pkl")
    EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)

    # Load documents (DataFrame)
    import pandas as pd
    df = pd.read_pickle(DATA_PICKLE_PATH)
    # Ensure 'combined_text' column exists
    if 'combined_text' not in df.columns:
        df['combined_text'] = df['Question']  # Use 'Question' column for embeddings

    docs = df['Answer'].tolist()  # Use corresponding 'Answer' column as the context

    # Load embedding model
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Initialize FastHTML app
    fasthtml_app, rt = fast_app(
        hdrs=(
            Script(src="https://cdn.tailwindcss.com"),
            Link(
                rel="stylesheet",
                href=(
                    "https://cdn.jsdelivr.net/npm/"
                    "daisyui@4.11.1/dist/full.min.css"
                ),
            ),
        ),
        ws_hdr=True,
    )

    @rt("/")
    async def get():
        return Div(
            H1(
                "Chat with Agony Aunt",
                cls="text-3xl font-bold mb-4 text-white"
            ),
            chat(),
            # Model status indicator
            Div(
                Span("Model status: "),
                Span("⚫", id="model-status-emoji"),
                cls="model-status text-white mt-4"
            ),
            Div(id="top-sources"),  # Placeholder for Top Sources
            cls="flex flex-col items-center min-h-screen bg-black",
        )

    # Placeholder implementation for arrow_circle_icon
    def arrow_circle_icon():
        # Replace this with your actual arrow_circle_icon implementation
        return Span("→")  # Placeholder

    def chat_top_sources(top_sources):
        return Div(
            Div(
                Div("Top Sources", cls="text-zinc-400 text-sm"),
                Div(
                    *[
                        A(
                            Span(
                                source['data_headline'][:21],
                                cls=(
                                    "font-mono text-green-500 group-hover:"
                                    "text-green-400 text-sm"
                                ),
                            ),
                            Div(
                                arrow_circle_icon(),
                                cls=(
                                    "flex items-center justify-center text-green-500"
                                    " group-hover:text-green-400 size-5 -rotate-45"
                                ),
                            ),
                            href=source['url'],
                            target="_blank",
                            cls=(
                                "justify-between items-center pl-2 pr-1 flex border"
                                " border-green-500 w-48 rounded-md group"
                            ),
                        )
                        for source in top_sources
                    ],
                    cls="flex items-center justify-start gap-2",
                ),
                cls="flex flex-col items-center gap-1",
            ),
            cls="flex flex-col items-center gap-1",
        )

    import aiohttp  # For asynchronous HTTP requests

    @fasthtml_app.ws("/ws")
    async def ws(msg: str, send):
        response_received = asyncio.Event()  # Event to indicate if response has been received

        # Increase max_tokens as per your request
        max_tokens = 6000

        async def update_model_status():
            # Wait for 3 seconds
            await asyncio.sleep(3)
            if not response_received.is_set():
                # Start switching between ⚫ and 🟡 every second
                for _ in range(20):  # 20 times * 2 seconds = 40 seconds
                    if response_received.is_set():
                        break
                    # Toggle to 🟡
                    await send(
                        Span(
                            "🟡",
                            id="model-status-emoji",
                            hx_swap_oob="innerHTML"
                        )
                    )
                    await asyncio.sleep(1)
                    if response_received.is_set():
                        break
                    # Toggle to ⚫
                    await send(
                        Span(
                            "⚫",
                            id="model-status-emoji",
                            hx_swap_oob="innerHTML"
                        )
                    )
                    await asyncio.sleep(1)
                else:
                    # After 40 seconds, set status to 🔴 if no response
                    if not response_received.is_set():
                        await send(
                            Span(
                                "🔴",
                                id="model-status-emoji",
                                hx_swap_oob="innerHTML"
                            )
                        )
            if response_received.is_set():
                # Set status to 🟢
                await send(
                    Span(
                        "🟢",
                        id="model-status-emoji",
                        hx_swap_oob="innerHTML"
                    )
                )
                # Wait for 10 minutes, then set back to ⚫
                await asyncio.sleep(600)
                await send(
                    Span(
                        "⚫",
                        id="model-status-emoji",
                        hx_swap_oob="innerHTML"
                    )
                )

        # Start the update_model_status coroutine
        asyncio.create_task(update_model_status())

        # Add user's message to chat history
        chat_messages.append({"role": "user", "content": msg})
        await send(chat_form(disabled=True))
        await send(
            Div(
                chat_message(len(chat_messages) - 1),
                id="messages",
                hx_swap_oob="beforeend"
            )
        )

        # Compute embedding of user's message
        question_embedding = emb_model.encode([msg], normalize_embeddings=True)
        question_embedding = question_embedding.astype('float32')

        # Retrieve top K similar documents
        K = 3
        distances, indices = index.search(question_embedding, K)
        retrieved_docs = [docs[idx] for idx in indices[0]]

        # Extract 'data-headline' and 'URL' of the top documents
        top_sources = []
        for idx in indices[0][:2]:  # Top 2 documents
            data_headline = df.iloc[idx]['data-headline']
            url = df.iloc[idx]['URL']  # Ensure 'URL' is a column in your DataFrame
            top_sources.append({'data_headline': data_headline, 'url': url})

        # Construct context from retrieved documents
        context = "\n\n".join(retrieved_docs)

        # Build conversation history (limit to last N messages to stay within context window)
        def count_tokens(text):
            return len(tokenizer.encode(text))

        # Build conversation history (limit to last N messages to stay within context window)
        def build_conversation(chat_messages, max_length=2000):
            conversation = ''
            total_length = 0
            # Start from the latest messages
            for message in reversed(chat_messages):
                role = message['role']
                content = message['content']
                message_text = f"{role.capitalize()}: {content}\n"
                total_length += len(message_text)
                if total_length > max_length:
                    break
                conversation = message_text + conversation
            return conversation

        conversation_history = build_conversation(chat_messages)

        # Simplified system prompt for better focus
        system_prompt = (
            "You are an Agony Aunt who helps individuals clarify their options and think through their choices. "
            "Provide thoughtful, empathetic, and helpful responses based on the user's concerns."
            "Review provided context for helpful information."
        )

        # Limit context to the most relevant snippet (e.g., the top result)
        context = retrieved_docs[0]  # Take only the most relevant retrieved document

        # Build the prompt by concatenating the system prompt, context, and conversation history
        prompt = (
            f"{system_prompt}\n\nContext Information:\n{context}\n\n"
            f"Conversation History:\n{conversation_history}\nAssistant:"
        )

        # Log the final prompt for debugging purposes
        print(f"Final Prompt being passed to the LLM:\n{prompt}\n")

        # Send prompt to vLLM server using aiohttp
        vllm_url = f"https://{USERNAME}--{APP_NAME}-serve-vllm.modal.run/v1/completions"
        params = {"prompt": prompt, "max_tokens": 2000, "stream": "true"}

        async with aiohttp.ClientSession() as session:
            async with session.get(vllm_url, params=params) as response:
                if response.status == 200:
                    # Indicate that response has been received
                    response_received.set()

                    # Add assistant's response to chat history
                    chat_messages.append({"role": "assistant", "content": ""})
                    message_index = len(chat_messages) - 1
                    await send(
                        Div(
                            chat_message(message_index),
                            id="messages",
                            hx_swap_oob="beforeend"
                        )
                    )

                    # Stream response to the user
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            text = chunk.decode('utf-8')
                            chat_messages[message_index]["content"] += text
                            await send(
                                Span(
                                    text,
                                    id=f"msg-content-{message_index}",
                                    hx_swap_oob="beforeend"
                                )
                            )
                else:
                    # Handle error
                    message = "Error: Unable to get response from LLM."
                    chat_messages.append({"role": "assistant", "content": message})
                    await send(
                        Div(
                            chat_message(len(chat_messages) - 1),
                            id="messages",
                            hx_swap_oob="beforeend"
                        )
                    )

        # Send the 'Top Sources' to the 'top-sources' div below the chat window
        await send(
            Div(
                chat_top_sources(top_sources),
                id="top-sources",
                hx_swap_oob="innerHTML"
            )
        )

        await send(chat_form(disabled=False))


    return fasthtml_app

if __name__ == "__main__":
    serve_vllm()      # Serve the vLLM server
    serve_fasthtml()  # Serve the FastHTML web interface
