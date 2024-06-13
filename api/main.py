from fastapi import FastAPI, APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Generator, Dict, Any
from phi.assistant import Assistant, AssistantMemory, AssistantKnowledge
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.memory.db.postgres import PgMemoryDb
from phi.embedder.openai import OpenAIEmbedder
from textwrap import dedent
import logging
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

db_url = "postgresql+psycopg://postgres.qsswdusttgzhprqgmaez:Burewala_789@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
storage = PgAssistantStorage(table_name="my_assistant", db_url=db_url)

app = FastAPI()
router = APIRouter()

# Configure CORS
origins = [
    "http://localhost:3000",  # Your Next.js frontend
    "http://127.0.0.1:3000",  # Your Next.js frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    stream: bool = False
    run_id: Optional[str] = None
    user_id: Optional[str] = "user"
    assistant: str = "RAG_PDF"
    new: bool = False

def get_assistant(run_id: Optional[str], user_id: Optional[str]) -> Assistant:
    assistant = Assistant(
        run_id=run_id,
        user_id=user_id,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
        # debug_mode=True,
        create_memories=True,
        memory=AssistantMemory(
            db=PgMemoryDb(
                db_url=db_url,
                table_name="personalized_assistant_memory",
            )
        ),
        update_memory_after_run=True,
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection="personalized_assistant_documents",
                embedder=OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536),
            ),
            num_documents=3,
        ),
        add_chat_history_to_messages=True,
        introduction=dedent(
            """\
            Hi, I'm your personalized Assistant called `OptimusV7`.
            I can remember details about your preferences and solve problems using tools and other AI Assistants.
            Let's get started!\
            """
        )
    )
    return assistant

def chat_response_streamer(assistant: Assistant, message: str) -> Generator:
    for chunk in assistant.run(message):
        yield chunk

@router.post("/chat")
async def chat(body: ChatRequest):
    """Sends a message to an Assistant and returns the response"""

    logger.debug(f"ChatRequest: {body}")
    run_id: Optional[str] = None

    if not body.new:
        existing_run_ids: List[str] = storage.get_all_run_ids(body.user_id)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant: Assistant = get_assistant(
        run_id=run_id, user_id=body.user_id
    )

    if body.stream:
        return StreamingResponse(
            chat_response_streamer(assistant, body.message),
            media_type="text/event-stream",
        )
    else:
        response = assistant.run(body.message, stream=False)
        return {"response": response}
    
class ChatHistoryRequest(BaseModel):
    run_id: str
    user_id: Optional[str] = None


@router.post("/history", response_model=List[Dict[str, Any]])
async def get_chat_history(body: ChatHistoryRequest):
    """Return the chat history for an Assistant run"""

    logger.debug(f"ChatHistoryRequest: {body}")
    assistant: Assistant = get_assistant(
        run_id=body.run_id, user_id=body.user_id
    )
    # Load the assistant from the database
    assistant.read_from_storage()

    chat_history = assistant.memory.get_chat_history()
    return chat_history

@router.get("/")
async def health_check():
    return "The health check is successful!"

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# import typer
# from rich.prompt import Prompt
# from typing import Optional, List
# from phi.assistant import Assistant, AssistantMemory, AssistantKnowledge
# from phi.storage.assistant.postgres import PgAssistantStorage
# from phi.knowledge.pdf import PDFUrlKnowledgeBase
# from phi.vectordb.pgvector import PgVector2
# from phi.memory.db.postgres import PgMemoryDb
# from phi.embedder.openai import OpenAIEmbedder
# from textwrap import dedent


# # db_url = "postgresql+psycopg://postgres.imfehfsoxnphvwikdgck:krpz7z$mhzQ3&)B@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"
# # db_url = "postgresql+psycopg://postgres.wkbicztcjyvpzjjvslif:Burewala_789@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
# db_url = "postgresql+psycopg://postgres.qsswdusttgzhprqgmaez:Burewala_789@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"

# # knowledge_base = PDFUrlKnowledgeBase(
# #     urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
# #     vector_db=PgVector2(collection="recipes", db_url=db_url),
# # )
# # Comment out after first run
# # knowledge_base.load()

# storage = PgAssistantStorage(table_name="my_assistant", db_url=db_url)


# def pdf_assistant(new: bool = False, user: str = "user"):
#     run_id: Optional[str] = None

#     if not new:
#         existing_run_ids: List[str] = storage.get_all_run_ids(user)
#         if len(existing_run_ids) > 0:
#             run_id = existing_run_ids[0]

#     assistant = Assistant(
#         run_id=run_id,
#         user_id=user,
#         # knowledge_base=knowledge_base,
#         storage=storage,
#         # Show tool calls in the response
#         show_tool_calls=True,
#         # Enable the assistant to search the knowledge base
#         search_knowledge=True,
#         # Enable the assistant to read the chat history
#         read_chat_history=True,

#         debug_mode=True,

#         create_memories= True,

#         memory= AssistantMemory(
#             db = PgMemoryDb(
#                 db_url=db_url,
#                 table_name="personalized_assistant_memory",
#             )
#         ),
#         # Update memory after each run
#         update_memory_after_run=True,
#          # Store knowledge in a vector database
#         knowledge_base=AssistantKnowledge(
#             vector_db=PgVector2(
#                 db_url=db_url,
#                 collection="personalized_assistant_documents",
#                 embedder=OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536),
#             ),
#             # 3 references are added to the prompt
#             num_documents=3,
#             ),
#         # This setting adds chat history to the messages
#         add_chat_history_to_messages=True,
        
#         introduction=dedent(
#             """\
#         Hi, I'm your personalized Assistant called `OptimusV7`.
#         I can remember details about your preferences and solve problems using tools and other AI Assistants.
#         Lets get started!\
#         """
#         )
#         )
    
#     if run_id is None:
#         run_id = assistant.run_id
#         print(f"Started Run: {run_id}\n")
#     else:
#         print(f"Continuing Run: {run_id}\n")

#     # Runs the assistant as a cli app
#     assistant.cli_app(markdown=True)


# if __name__ == "__main__":
#     typer.run(pdf_assistant)
