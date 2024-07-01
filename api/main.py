# OpneAI LLM:
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Generator, Dict, Any
from phi.assistant import Assistant, AssistantMemory, AssistantKnowledge, AssistantRun
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.memory.db.postgres import PgMemoryDb
from phi.embedder.openai import OpenAIEmbedder
from textwrap import dedent
import logging
from fastapi.middleware.cors import CORSMiddleware
from phi.tools.duckduckgo import DuckDuckGo
from sqlalchemy import text
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json
from typing import List, Dict, Any


from contextlib import contextmanager

logger = logging.getLogger(__name__)

class CustomPgAssistantStorage(PgAssistantStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)

    def get_all_run_info(self, user_id: str) -> List[Dict[str, Any]]:
        run_ids = self.get_all_run_ids(user_id)
        run_info_list = []

        query = text(f"""
        SELECT run_id, assistant_data, run_data, memory
        FROM {self.schema}.{self.table_name}
        WHERE run_id = ANY(:run_ids)
        ORDER BY created_at DESC
        """)

        with self.Session() as session:
            result = session.execute(query, {"run_ids": run_ids})
            rows = result.fetchall()

            for row in rows:
                run_id, assistant_data, run_data, memory = row

                # Handle cases where data might be string or dict
                if isinstance(assistant_data, str):
                    assistant_data = json.loads(assistant_data)
                elif assistant_data is None:
                    assistant_data = {}
                
                if isinstance(run_data, str):
                    run_data = json.loads(run_data)
                elif run_data is None:
                    run_data = {}

                if isinstance(memory, str):
                    memory = json.loads(memory)
                elif memory is None:
                    memory = {}

                chat_history = memory.get('chat_history', [])

                # Get the last response from chat_history where role is 'assistant'
                last_response = None
                for entry in reversed(chat_history):
                    if entry['role'] == 'assistant':
                        last_response = entry['content']
                        break

                run_info = {
                    'run_id': run_id,
                    'template_id': assistant_data.get('template_id') or run_data.get('template_id'),
                    'template_title': assistant_data.get('template_title') or run_data.get('template_title'),
                    'template_data': assistant_data,
                    'last_response': last_response  # Add last_response to the run_info
                }
                run_info_list.append(run_info)

        return run_info_list
    
# Initialize the custom storage
db_url = "postgresql+psycopg://postgres.qsswdusttgzhprqgmaez:Burewala_789@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
storage = CustomPgAssistantStorage(table_name="my_assistant", db_url=db_url)


app = FastAPI()
router = APIRouter()

# Configure CORS
origins = [
    "http://localhost:3000",  
    "http://127.0.0.1:3000",  
    "https://app.kyndom.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_dynamic_instructions(template_type):
    if template_type == "REELS_IDEAS":
        specific_instructions = [
            "1. Understand the Template's Purpose and Elements: Familiarize yourself with the purpose of the template and the key elements it contains.",
            "2. Collect Data: Identify the user's preferences and any custom variables in the template. Based on the user's preferences, determine if additional information is needed to complete the custom variables. Ask the user one question at a time to gather the necessary data. Wait for the user's response before asking the next question. Limit the total number of questions to a maximum of 5.",
            "3. If Needed, Search from the Web: Utilize external sources like DuckDuckGo to gather additional information if required.",
            "4. Personalize the Reel Idea: Use the collected data my saved profile information to personalize the reel idea according to the user's preferences and the template's requirements.",
            "5. Tone: Ensure the tone of the personalized template is selected by the user.",
            "6. Format: Maintain the original format of the template.",
            "7. Length: Ensure the personalized template does not exceed 350 words.",
        ]
    elif template_type == "STORY_IDEAS":
        specific_instructions = [
            "1. Understand the Template's Purpose and Elements: Familiarize yourself with the purpose of the template and the key elements it contains.",
            "2. Collect Data: Identify the user's preferences and any custom variables in the template. Based on the user's preferences, determine if additional information is needed to complete the custom variables. Ask the user one question at a time to gather the necessary data. Wait for the user's response before asking the next question. Limit the total number of questions to a maximum of 5.",
            "3. If Needed, Search from the Web: Utilize external sources like DuckDuckGo to gather additional information if required.",
            "4. Personalize the Story Idea: Use the collected data my saved profile information to personalize the story idea according to the user's preferences and the template's requirements.",
            "5. Tone: Ensure the tone of the personalized story idea is selected by the user.",
            "6. Format: Maintain the original format of the story idea.",
            "7. Length: Ensure the personalized story idea does not exceed 350 words."
        ]
    elif template_type == "TODAYS_PLAN":
        specific_instructions = [
            "This is a 'Today's Plan' template.",
            "1. Understand the Template's Purpose and Elements: Familiarize yourself with the purpose of the template and the key elements it contains.",
            "2. Collect Data: Identify the user's preferences and any custom variables in the template. Based on the user's preferences, determine if additional information is needed to complete the custom variables. Ask the user one question at a time to gather the necessary data. Wait for the user's response before asking the next question. Limit the total number of questions to a maximum of 5.",
            "3. If Needed, Search from the Web: Utilize external sources like DuckDuckGo to gather additional information if required.",
            "4. Personalize the Idea: Use the collected data and my saved profile information to personalize the idea according to the user's preferences and the template's requirements.",
            "5. Tone: Ensure the tone of the personalized idea is selected by the user.",
            "6. Format: Maintain the original format of the idea.",
            "7. Length: Ensure the personalized idea does not exceed 350 words."
        ]
    else:
        specific_instructions = []

    return specific_instructions

class ChatRequest(BaseModel):
    message: str
    stream: bool = False
    run_id: Optional[str] = None
    user_id: Optional[str] = "user"
    assistant: str = "RAG_PDF"
    new: bool = False
    template_type: Optional[str] = None
    template_title: Optional[str] = None  # New field
    template_id: Optional[str] = None  # New field

def get_assistant(run_id: Optional[str], user_id: Optional[str], template_type: Optional[str], template_title: Optional[str] = None, template_id: Optional[str] = None) -> Assistant:
    assistant_params = {
        "description": "You are a real estate assistant for my real estate agency",
        "run_id": run_id,
        "user_id": user_id,
        "storage": storage,
        "tools": [DuckDuckGo()],
        "search_knowledge": True,
        "read_chat_history": True,
        "create_memories": True,
        "memory": AssistantMemory(
            db=PgMemoryDb(
                db_url=db_url,
                table_name="personalized_assistant_memory",
            )
        ),
        "update_memory_after_run": True,
        "knowledge_base": AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection="personalized_assistant_documents",
                embedder=OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536),
            ),
            num_documents=3,
        ),
        "add_chat_history_to_messages": True,
        "introduction": dedent(
            """\
            Hi, I'm your personalized Assistant called OptimusV7.
            I can remember details about your preferences and solve problems using tools and other AI Assistants.
            Let's get started!\
            """
        ),
        "assistant_data": {
            "template_title": template_title,
            "template_id": template_id
        }
    }

    if template_type:
        assistant_params["instructions"] = get_dynamic_instructions(template_type)

    assistant = Assistant(**assistant_params)
    return assistant

def chat_response_streamer(assistant: Assistant, message: str, is_new_session: bool) -> Generator:
    if is_new_session:
        yield f"run_id: {assistant.run_id}\n"  # Yield the run_id first for new sessions
    for chunk in assistant.run(message):
        yield chunk
    yield "[DONE]\n\n"

@router.post("/chat")
async def chat(body: ChatRequest):
    """Sends a message to an Assistant and returns the response"""
    
    logger.debug(f"ChatRequest: {body}")
    run_id: Optional[str] = None
    is_new_session = False
    
    if body.new:
        is_new_session = True
    else:
        existing_run_ids: List[str] = storage.get_all_run_ids(body.user_id)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]
    
    assistant: Assistant = get_assistant(
        run_id=run_id,
        user_id=body.user_id,
        template_type=body.template_type,
        template_title=body.template_title,
        template_id=body.template_id
    )
    
    if body.stream:
        return StreamingResponse(
            chat_response_streamer(assistant, body.message, is_new_session),
            media_type="text/event-stream",
        )
    else:
        response = assistant.run(body.message, stream=False)
        if is_new_session:
            return JSONResponse({"run_id": assistant.run_id, "response": response})
        else:
            return JSONResponse({"response": response})
    
class ChatHistoryRequest(BaseModel):
    run_id: str
    user_id: Optional[str] = None


@router.post("/history", response_model=List[Dict[str, Any]])
async def get_chat_history(body: ChatHistoryRequest):
    """Return the chat history for an Assistant run"""

    logger.debug(f"ChatHistoryRequest: {body}")
    assistant: Assistant = get_assistant(
        run_id=body.run_id, user_id=body.user_id, template_type=None
    )
    # Load the assistant from the database
    assistant.read_from_storage()

    chat_history = assistant.memory.get_chat_history()
    return chat_history

@router.get("/")
async def health_check():
    return "The health check is successful!"

class GetAllAssistantRunsRequest(BaseModel):
    user_id: str

@app.post("/get-all", response_model=List[AssistantRun])
def get_assistants(body: GetAllAssistantRunsRequest):
    """Return all Assistant runs for a user"""
    return storage.get_all_runs(user_id=body.user_id)

class RunInfo(BaseModel):
    run_id: str
    template_id: Optional[str] = None
    template_title: Optional[str] = None
    last_response: Optional[str] = None

class GetAllAssistantRunIdsRequest(BaseModel):
    user_id: str

@app.post("/get-all-ids", response_model=List[RunInfo])
def get_run_ids(body: GetAllAssistantRunIdsRequest):
    """Return all run_ids with template info for a user"""
    try:
        run_info_list = storage.get_all_run_info(user_id=body.user_id)
        return [RunInfo(**run_info) for run_info in run_info_list]
    except Exception as e:
        logger.exception("An error occurred in get_run_ids")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# # OpneAI LLM:
# from fastapi import FastAPI, APIRouter
# from fastapi.responses import StreamingResponse, JSONResponse
# from pydantic import BaseModel
# from typing import Optional, List, Generator, Dict, Any
# from phi.assistant import Assistant, AssistantMemory, AssistantKnowledge, AssistantRun
# from phi.storage.assistant.postgres import PgAssistantStorage
# from phi.knowledge.pdf import PDFUrlKnowledgeBase
# from phi.vectordb.pgvector import PgVector2
# from phi.memory.db.postgres import PgMemoryDb
# from phi.embedder.openai import OpenAIEmbedder
# from textwrap import dedent
# import logging
# from fastapi.middleware.cors import CORSMiddleware
# from phi.tools.duckduckgo import DuckDuckGo
# import json

# logger = logging.getLogger(__name__)

# db_url = "postgresql+psycopg://postgres.qsswdusttgzhprqgmaez:Burewala_789@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
# storage = PgAssistantStorage(table_name="my_assistant", db_url=db_url)

# app = FastAPI()
# router = APIRouter()

# # Configure CORS
# origins = [
#     "http://localhost:3000",  
#     "http://127.0.0.1:3000",  
#     "https://app.kyndom.com"
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def get_dynamic_instructions(template_type):
#     if template_type == "REELS_IDEAS":
#         specific_instructions = [
#             "1. Understand the Template's Purpose and Elements: Familiarize yourself with the purpose of the template and the key elements it contains.",
#             "2. Collect Data: Identify the user's preferences and any custom variables in the template. Based on the user's preferences, determine if additional information is needed to complete the custom variables. Ask the user one question at a time to gather the necessary data. Wait for the user's response before asking the next question. Limit the total number of questions to a maximum of 5.",
#             "3. If Needed, Search from the Web: Utilize external sources like DuckDuckGo to gather additional information if required.",
#             "4. Personalize the Reel Idea: Use the collected data my saved profile information to personalize the reel idea according to the user's preferences and the template's requirements.",
#             "5. Tone: Ensure the tone of the personalized template is selected by the user.",
#             "6. Format: Maintain the original format of the template.",
#             "7. Length: Ensure the personalized template does not exceed 350 words.",
#         ]
#     # "2. Do Not Use Previous Reel Idea Information: Do not use the information from previous Reel Ideas for the current Reel Idea. Each Reel Idea should be treated independently.",
#     elif template_type == "STORY_IDEAS":
#         specific_instructions = [
#             "1. Understand the Template's Purpose and Elements: Familiarize yourself with the purpose of the template and the key elements it contains.",
#             "2. Collect Data: Identify the user's preferences and any custom variables in the template. Based on the user's preferences, determine if additional information is needed to complete the custom variables. Ask the user one question at a time to gather the necessary data. Wait for the user's response before asking the next question. Limit the total number of questions to a maximum of 5.",
#             "3. If Needed, Search from the Web: Utilize external sources like DuckDuckGo to gather additional information if required.",
#             "4. Personalize the Story Idea: Use the collected data my saved profile information to personalize the story idea according to the user's preferences and the template's requirements.",
#             "5. Tone: Ensure the tone of the personalized story idea is selected by the user.",
#             "6. Format: Maintain the original format of the story idea.",
#             "7. Length: Ensure the personalized story idea does not exceed 350 words."
#         ]
#     # "2. Do Not Use Previous Information: Do not use the information from previous templates for the current template. Each template should be treated independently.",
#     elif template_type == "TODAYS_PLAN":
#         specific_instructions = [
#             "This is a 'Today's Plan' template.",
#             "1. Understand the Template's Purpose and Elements: Familiarize yourself with the purpose of the template and the key elements it contains.",
#             "2. Collect Data: Identify the user's preferences and any custom variables in the template. Based on the user's preferences, determine if additional information is needed to complete the custom variables. Ask the user one question at a time to gather the necessary data. Wait for the user's response before asking the next question. Limit the total number of questions to a maximum of 5.",
#             "3. If Needed, Search from the Web: Utilize external sources like DuckDuckGo to gather additional information if required.",
#             "4. Personalize the Idea: Use the collected data and my saved profile information to personalize the idea according to the user's preferences and the template's requirements.",
#             "5. Tone: Ensure the tone of the personalized idea is selected by the user.",
#             "6. Format: Maintain the original format of the idea.",
#             "7. Length: Ensure the personalized idea does not exceed 350 words."
#         ]
#     # "2. Do Not Use Previous Information: Do not use the information from previous templates for the current template. Each template should be treated independently.",
#     else:
#         specific_instructions = []

#     return specific_instructions

# class ChatRequest(BaseModel):
#     message: str
#     stream: bool = False
#     run_id: Optional[str] = None
#     user_id: Optional[str] = "user"
#     assistant: str = "RAG_PDF"
#     new: bool = False
#     template_type: Optional[str] = None

# def get_assistant(run_id: Optional[str], user_id: Optional[str], template_type: Optional[str]) -> Assistant:
#     assistant_params = {
#         "description": "You are a real estate assistant for my real estate agency",
#         "run_id": run_id,
#         "user_id": user_id,
#         "storage": storage,
#         "tools": [DuckDuckGo()],
#         "show_tool_calls": True,
#         "search_knowledge": True,
#         "read_chat_history": True,
#         # "debug_mode": True,
#         "create_memories": True,
#         "memory": AssistantMemory(
#             db=PgMemoryDb(
#                 db_url=db_url,
#                 table_name="personalized_assistant_memory",
#             )
#         ),
#         "update_memory_after_run": True,
#         "knowledge_base": AssistantKnowledge(
#             vector_db=PgVector2(
#                 db_url=db_url,
#                 collection="personalized_assistant_documents",
#                 embedder=OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536),
#             ),
#             num_documents=3,
#         ),
#         "add_chat_history_to_messages": True,
#         "introduction": dedent(
#             """\
#             Hi, I'm your personalized Assistant called OptimusV7.
#             I can remember details about your preferences and solve problems using tools and other AI Assistants.
#             Let's get started!\
#             """
#         )
#     }

#     if template_type:
#         assistant_params["instructions"] = get_dynamic_instructions(template_type)

#     assistant = Assistant(**assistant_params)
#     return assistant


# # def get_assistant(run_id: Optional[str], user_id: Optional[str], template_type: Optional[str]) -> Assistant:
# #     assistant = Assistant(
# #         description="You are a real estate assistant for my real estate agency",
# #         instructions= get_dynamic_instructions(template_type),
# #         run_id=run_id,
# #         user_id=user_id,
# #         storage=storage,
# #         tools=[DuckDuckGo()],
# #         show_tool_calls=True,
# #         search_knowledge=True,
# #         read_chat_history=True,
# #         debug_mode=True,
# #         create_memories=True,
# #         memory=AssistantMemory(
# #             db=PgMemoryDb(
# #                 db_url=db_url,
# #                 table_name="personalized_assistant_memory",
# #             )
# #         ),
# #         update_memory_after_run=True,
# #         knowledge_base=AssistantKnowledge(
# #             vector_db=PgVector2(
# #                 db_url=db_url,
# #                 collection="personalized_assistant_documents",
# #                 embedder=OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536),
# #             ),
# #             num_documents=3,
# #         ),
# #         add_chat_history_to_messages=True,
# #         introduction=dedent(
# #             """\
# #             Hi, I'm your personalized Assistant called OptimusV7.
# #             I can remember details about your preferences and solve problems using tools and other AI Assistants.
# #             Let's get started!\
# #             """
# #         )
# #     )
# #     return assistant


# def chat_response_streamer(assistant: Assistant, message: str, is_new_session: bool) -> Generator:
#     if is_new_session:
#         yield f"run_id: {assistant.run_id}\n"  # Yield the run_id first for new sessions
#     for chunk in assistant.run(message):
#         yield chunk
#     yield "[DONE]\n\n"

# @router.post("/chat")
# async def chat(body: ChatRequest):
#     """Sends a message to an Assistant and returns the response"""
    
#     logger.debug(f"ChatRequest: {body}")
#     run_id: Optional[str] = None
#     is_new_session = False
    
#     if body.new:
#         is_new_session = True  
#     else:
#         existing_run_ids: List[str] = storage.get_all_run_ids(body.user_id)
#         if len(existing_run_ids) > 0:
#             run_id = existing_run_ids[0]
    
#     assistant: Assistant = get_assistant(
#         run_id=run_id, user_id=body.user_id, template_type=body.template_type
#     )
    
#     if body.stream:
#         return StreamingResponse(
#             chat_response_streamer(assistant, body.message, is_new_session),
#             media_type="text/event-stream",
#         )
#     else:
#         response = assistant.run(body.message, stream=False)
#         if is_new_session:
#             return JSONResponse({"run_id": assistant.run_id, "response": response})
#         else:
#             return JSONResponse({"response": response})
    
# class ChatHistoryRequest(BaseModel):
#     run_id: str
#     user_id: Optional[str] = None


# @router.post("/history", response_model=List[Dict[str, Any]])
# async def get_chat_history(body: ChatHistoryRequest):
#     """Return the chat history for an Assistant run"""

#     logger.debug(f"ChatHistoryRequest: {body}")
#     assistant: Assistant = get_assistant(
#         run_id=body.run_id, user_id=body.user_id, template_type=None
#     )
#     # Load the assistant from the database
#     assistant.read_from_storage()

#     chat_history = assistant.memory.get_chat_history()
#     return chat_history

# @router.get("/")
# async def health_check():
#     return "The health check is successful!"

# class GetAllAssistantRunsRequest(BaseModel):
#     user_id: str

# @app.post("/get-all", response_model=List[AssistantRun])
# def get_assistants(body: GetAllAssistantRunsRequest):
#     """Return all Assistant runs for a user"""
#     return storage.get_all_runs(user_id=body.user_id)


# class GetAllAssistantRunIdsRequest(BaseModel):
#     user_id: str

# @app.post("/get-all-ids", response_model=List[str])
# def get_run_ids(body: GetAllAssistantRunIdsRequest):
#     """Return all run_ids for a user"""
#     return storage.get_all_run_ids(user_id=body.user_id)


# app.include_router(router)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



    

#####################################################################

# Gemini LLM
# import os
# import logging
# from fastapi import FastAPI, APIRouter
# from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Optional, List, Generator, Dict, Any
# from phi.assistant import Assistant, AssistantMemory, AssistantKnowledge
# from phi.storage.assistant.postgres import PgAssistantStorage
# from phi.vectordb.pgvector import PgVector2
# from phi.memory.db.postgres import PgMemoryDb
# from textwrap import dedent
# from phi.llm.gemini import Gemini
# import vertexai

# logger = logging.getLogger(__name__)

# # Set environment variables for Gemini
# # os.environ['PROJECT_ID'] = 'Generative Language Client'
# # os.environ['LOCATION'] = 'Canada'

# # Initialize VertexAI
# vertexai.init(project=os.getenv("PROJECT_ID"), location=os.getenv("LOCATION"))

# db_url = "postgresql+psycopg://postgres.qsswdusttgzhprqgmaez:Burewala_789@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
# storage = PgAssistantStorage(table_name="my_assistant", db_url=db_url)

# app = FastAPI()
# router = APIRouter()

# # Configure CORS
# origins = [
#     "http://localhost:3000",  # Your Next.js frontend
#     "http://localhost:3001",  # Your Next.js frontend
#     "http://127.0.0.1:3001",  # Your Next.js frontend
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatRequest(BaseModel):
#     message: str
#     stream: bool = False
#     run_id: Optional[str] = None
#     user_id: Optional[str] = "user"
#     assistant: str = "RAG_PDF"
#     new: bool = False

# def get_assistant(run_id: Optional[str], user_id: Optional[str]) -> Assistant:
#     assistant = Assistant(
#         description="You are a real estate assistant for my real estate agency",
#         instructions=[""]
#         llm=Gemini(
#             model="gemini-1.0-pro-vision"
#         ),
#         run_id=run_id,
#         user_id=user_id,
#         storage=storage,
#         # show_tool_calls=True,
#         search_knowledge=True,
#         read_chat_history=True,
#         create_memories=True,
#         memory=AssistantMemory(
#             db=PgMemoryDb(
#                 db_url=db_url,
#                 table_name="personalized_assistant_memory",
#             )
#         ),
#         update_memory_after_run=True,
#         knowledge_base=AssistantKnowledge(
#             vector_db=PgVector2(
#                 db_url=db_url,
#                 collection="personalized_assistant_documents",
#                 # embedder=Gemini(model="gemini-1.0-pro-vision"),  # Using Gemini model
#             ),
#             num_documents=3,
#         ),
#         add_chat_history_to_messages=True,
#         introduction=dedent(
#             """\
#             Hi, I'm your personalized Assistant called OptimusV7.
#             I can remember details about your preferences and solve problems using tools and other AI Assistants.
#             Let's get started!\
#             """
#         )
#     )
#     return assistant

# def chat_response_streamer(assistant: Assistant, message: str) -> Generator:
#     for chunk in assistant.run(message):
#         yield chunk

# @router.post("/chat")
# async def chat(body: ChatRequest):
#     """Sends a message to an Assistant and returns the response"""

#     logger.debug(f"ChatRequest: {body}")
#     run_id: Optional[str] = None

#     if not body.new:
#         existing_run_ids: List[str] = storage.get_all_run_ids(body.user_id)
#         if len(existing_run_ids) > 0:
#             run_id = existing_run_ids[0]

#     assistant: Assistant = get_assistant(
#         run_id=run_id, user_id=body.user_id
#     )

#     if body.stream:
#         return StreamingResponse(
#             chat_response_streamer(assistant, body.message),
#             media_type="text/event-stream",
#         )
#     else:
#         response = assistant.run(body.message, stream=False)
#         return {"response": response}
    
# class ChatHistoryRequest(BaseModel):
#     run_id: str
#     user_id: Optional[str] = None

# @router.post("/history", response_model=List[Dict[str, Any]])
# async def get_chat_history(body: ChatHistoryRequest):
#     """Return the chat history for an Assistant run"""

#     logger.debug(f"ChatHistoryRequest: {body}")
#     assistant: Assistant = get_assistant(
#         run_id=body.run_id, user_id=body.user_id
#     )
#     # Load the assistant from the database
#     assistant.read_from_storage()

#     chat_history = assistant.memory.get_chat_history()
#     return chat_history

# @router.get("/")
# async def health_check():
#     return "The health check is successful!"

# app.include_router(router)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)





# Claude LLM:
# from fastapi import FastAPI, APIRouter
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# from typing import Optional, List, Generator, Dict, Any
# from phi.assistant import Assistant, AssistantMemory
# from phi.storage.assistant.postgres import PgAssistantStorage
# from phi.memory.db.postgres import PgMemoryDb
# from phi.llm.anthropic import Claude
# from textwrap import dedent
# import logging
# from fastapi.middleware.cors import CORSMiddleware
# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# logger = logging.getLogger(__name__)

# # Ensure the API key is set
# api_key = os.getenv("ANTHROPIC_API_KEY")
# if not api_key:
#     raise ValueError("The ANTHROPIC_API_KEY environment variable is not set.")
# else:
#     logger.info(f"API Key: {api_key}")  # Print the key to ensure it's being read (remove this in production)

# db_url = "postgresql+psycopg://postgres.qsswdusttgzhprqgmaez:Burewala_789@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
# storage = PgAssistantStorage(table_name="my_assistant", db_url=db_url)

# app = FastAPI()
# router = APIRouter()

# # Configure CORS
# origins = [
#     "http://localhost:3000",  # Your Next.js frontend
#     "http://127.0.0.1:3000",  # Your Next.js frontend
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatRequest(BaseModel):
#     message: str
#     stream: bool = False
#     run_id: Optional[str] = None
#     user_id: Optional[str] = "user"
#     assistant: str = "RAG_PDF"
#     new: bool = False

# def get_assistant(run_id: Optional[str], user_id: Optional[str]) -> Assistant:
#     assistant = Assistant(
#         run_id=run_id,
#         user_id=user_id,
#         storage=storage,
#         show_tool_calls=True,
#         search_knowledge=True,
#         read_chat_history=True,
#         create_memories=True,
#         memory=AssistantMemory(
#             db=PgMemoryDb(
#                 db_url=db_url,
#                 table_name="personalized_assistant_memory",
#             )
#         ),
#         update_memory_after_run=True,
#         add_chat_history_to_messages=True,
#         llm=Claude(
#             model="claude-3-opus-20240229",
#             max_tokens=1024,
#             api_key=api_key,
#         ),
#         introduction=dedent(
#             """\
#             Hi, I'm your personalized Assistant called `OptimusV7`.
#             I can remember details about your preferences and solve problems using tools and other AI Assistants.
#             Let's get started!\
#             """
#         )
#     )
#     return assistant

# def chat_response_streamer(assistant: Assistant, message: str) -> Generator:
#     for chunk in assistant.run(message):
#         yield chunk

# @router.post("/chat")
# async def chat(body: ChatRequest):
#     """Sends a message to an Assistant and returns the response"""

#     logger.debug(f"ChatRequest: {body}")
#     run_id: Optional[str] = None

#     if not body.new:
#         existing_run_ids: List[str] = storage.get_all_run_ids(body.user_id)
#         if len(existing_run_ids) > 0:
#             run_id = existing_run_ids[0]

#     assistant: Assistant = get_assistant(
#         run_id=run_id, user_id=body.user_id
#     )

#     if body.stream:
#         return StreamingResponse(
#             chat_response_streamer(assistant, body.message),
#             media_type="text/event-stream",
#         )
#     else:
#         response = assistant.run(body.message, stream=False)
#         return {"response": response}
    
# class ChatHistoryRequest(BaseModel):
#     run_id: str
#     user_id: Optional[str] = None

# @router.post("/history", response_model=List[Dict[str, Any]])
# async def get_chat_history(body: ChatHistoryRequest):
#     """Return the chat history for an Assistant run"""

#     logger.debug(f"ChatHistoryRequest: {body}")
#     assistant: Assistant = get_assistant(
#         run_id=body.run_id, user_id=body.user_id
#     )
#     # Load the assistant from the database
#     assistant.read_from_storage()

#     chat_history = assistant.memory.get_chat_history()
#     return chat_history

# @router.get("/")
# async def health_check():
#     return "The health check is successful!"

# app.include_router(router)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# from fastapi import FastAPI, APIRouter
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# from typing import Optional, List, Generator, Dict, Any
# from phi.assistant import Assistant, AssistantMemory, AssistantKnowledge
# from phi.storage.assistant.postgres import PgAssistantStorage
# from phi.knowledge.pdf import PDFUrlKnowledgeBase
# from phi.vectordb.pgvector import PgVector2
# from phi.memory.db.postgres import PgMemoryDb
# from phi.embedder.openai import OpenAIEmbedder
# from textwrap import dedent
# import logging
# from fastapi.middleware.cors import CORSMiddleware
# from phi.llm.openai import OpenAIChat
# from phi.llm.anthropic import Claude

# logger = logging.getLogger(__name__)

# db_url = "postgresql+psycopg://postgres.qsswdusttgzhprqgmaez:Burewala_789@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"
# storage = PgAssistantStorage(table_name="my_assistant", db_url=db_url)

# app = FastAPI()
# router = APIRouter()

# # Configure CORS
# origins = [
#     "http://localhost:3000",  # Your Next.js frontend
#     "http://127.0.0.1:3000",  # Your Next.js frontend
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatRequest(BaseModel):
#     message: str
#     stream: bool = False
#     run_id: Optional[str] = None
#     user_id: Optional[str] = "user"
#     assistant: str = "RAG_PDF"
#     new: bool = False

# def get_assistant(run_id: Optional[str], user_id: Optional[str]) -> Assistant:
#     assistant = Assistant(
#         # llm=OpenAIChat(model="gpt-4-turbo", max_tokens=500, temperature=0.3),
#         llm=Claude(
#             model="claude-3-opus-20240229",
#             max_tokens=1024,
#         ),
#         run_id=run_id,
#         user_id=user_id,
#         storage=storage,
#         show_tool_calls=True,
#         search_knowledge=True,
#         read_chat_history=True,
#         # debug_mode=True,
#         create_memories=True,
#         # memory=AssistantMemory(
#         #     db=PgMemoryDb(
#         #         db_url=db_url,
#         #         table_name="personalized_assistant_memory",
#         #     )
#         # ),
#         update_memory_after_run=True,
#         knowledge_base=AssistantKnowledge(
#             vector_db=PgVector2(
#                 db_url=db_url,
#                 collection="personalized_assistant_documents",
#                 # embedder=OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536),
#                 embedder=Claude(model="claude-3-opus-20240229", max_tokens=1024),
#             ),
#             num_documents=3,
#         ),
#         add_chat_history_to_messages=True,
#         introduction=dedent(
#             """\
#             Hi, I'm your personalized Assistant called `OptimusV7`.
#             I can remember details about your preferences and solve problems using tools and other AI Assistants.
#             Let's get started!\
#             """
#         )
#     )
#     return assistant

# def chat_response_streamer(assistant: Assistant, message: str) -> Generator:
#     for chunk in assistant.run(message):
#         yield chunk

# @router.post("/chat")
# async def chat(body: ChatRequest):
#     """Sends a message to an Assistant and returns the response"""

#     logger.debug(f"ChatRequest: {body}")
#     run_id: Optional[str] = None

#     if not body.new:
#         existing_run_ids: List[str] = storage.get_all_run_ids(body.user_id)
#         if len(existing_run_ids) > 0:
#             run_id = existing_run_ids[0]

#     assistant: Assistant = get_assistant(
#         run_id=run_id, user_id=body.user_id
#     )

#     if body.stream:
#         return StreamingResponse(
#             chat_response_streamer(assistant, body.message),
#             media_type="text/event-stream",
#         )
#     else:
#         response = assistant.run(body.message, stream=False)
#         return {"response": response}
    
# class ChatHistoryRequest(BaseModel):
#     run_id: str
#     user_id: Optional[str] = None


# @router.post("/history", response_model=List[Dict[str, Any]])
# async def get_chat_history(body: ChatHistoryRequest):
#     """Return the chat history for an Assistant run"""

#     logger.debug(f"ChatHistoryRequest: {body}")
#     assistant: Assistant = get_assistant(
#         run_id=body.run_id, user_id=body.user_id
#     )
#     # Load the assistant from the database
#     assistant.read_from_storage()

#     chat_history = assistant.memory.get_chat_history()
#     return chat_history

# @router.get("/")
# async def health_check():
#     return "The health check is successful!"

# app.include_router(router)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)






















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
