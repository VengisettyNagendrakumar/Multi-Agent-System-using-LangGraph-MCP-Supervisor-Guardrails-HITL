import os 
import certifi
from dotenv import load_dotenv
import asyncio
load_dotenv()

os.environ["SSL_CERT_FILE"]=certifi.where()#certifi.where() returns the path to the certificate bundle installed with the certifi package. That bundle contains trusted Certificate Authority (CA) certificates.
os.environ["REQUEST_CA_BUNDLE"]=certifi.where()#SSL_CERT_FILE and REQUEST_CA_BUNDLE tell SSL/networking tools where to find trusted certificates when checking a website’s HTTPS certificate.
#Those lines force Python’s HTTPS libraries to use certifi’s trusted-certificate file when connecting to the flight API:

from typing import Annotated,TypedDict,Any
import operator 
from langgraph.types import Command, interrupt
import uuid 
import json
import psycopg 
from psycopg.rows import dict_row 
from langgraph.graph import StateGraph,START,END 
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage,HumanMessage,AIMessage,SystemMessage

from mcp_client import tavily_mcp_search,aviation_mcp_call,extract_destination,weather_mcp_seacrh,forecast_mcp_search

def get_database_url():
    database_url=os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing"
        )
    if "sllmode=" not in database_url:
        separator="&" if "?" in database_url else "?"
        database_url=f"{database_url}{separator}sslmode=require"#sslmode=require tells a PostgreSQL client to use an encrypted SSL/TLS connection. It protects data traveling between your app and the database from being read in transit.
    return database_url

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
    "GROQ_API_KEY is missing"
    )
llm=ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    max_tokens=1024
)

#state 
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]#accrding to prompt supervisor agent will select the agent 
    trip_constraints: dict[str, Any] #like user says from A TO B in budget 2 lakhs so it should extract source destination and budget adn store
    supervisor_reasoning: str #why selected that particular agent 

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New budget + HITL state
    budget_results: str
    approval_request: str #this is for HITL 
    approved: bool#tru or fasle like whether accepted or rejected 
    human_feedback: str#if he rejcted why 
    final_response: str

    llm_calls: int
    
#shared helpers
KNOWN_AGENTS={
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent"
}
AGENT_ORDER=[
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent"
]
    
def _llm_text(system_prompt:str,user_prompt:str)->str:
    response=llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
    )
    return str(response.content)
def _json_from_llm(text:str)->dict[str,Any]:
    "extract the first complete json object returned by the model"
    start=text.find("{")#• find(): Searches the string from left to right and returns the index of the first (earliest) occurrence of the specified substring.
#rfind(): Searches the string from right to left (reverse find) and returns the index of the last (highest) occurrence of the specified
    end=text.rfind("}")
    if start==-1 or end==-1 or end<start:
        raise ValueError("the model did not return a json object")
    return json.loads(text[start:end+1])
def _empty_constraints()->dict[str,Any]:
    return{
        "destination": "", #extract these from user query
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }

#gaurdrials will be 2 types one will be model based and second one is determinitistic we are using below model based that means we will use llm deterministic means like having a list of keywords and block them if they come in prompt
#supervisor agent + Input Guardrail 
def supervisor_agent(state:TravelState):
    query=state["user_query"]
    llm_calls=state.get("llm_calls",0)
    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true or false,
  "reason": ""
}}

User request:
{query}
"""
    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw=_llm_text(
             "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result=_json_from_llm(guardrail_raw)
        allowed=bool(guardrail_result.get("allowed",True))
        guardrail_reason=str(guardrail_result.get("reason", "")).strip()
        llm_calls+1
    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."
    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }
    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""
    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip() #why supervisor agent selected that particular tools
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }
# Guardrail blocked response
def guardrail_blocked_agent(state:TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }



#flight agent
# Flight Tool Router Prompt
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""




# Flight Agent
def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]

    try:

        airports = asyncio.run(
            aviation_mcp_call(
                "list_airports"
            )
        )

        airlines = asyncio.run(
            aviation_mcp_call(
                "list_airlines"
            )
        )


        print("\nAIRPORTS:", airports)
        print("\nAIRLINES:", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000]
        )

        response = llm.invoke([
            SystemMessage(
                content="You are an expert travel flight planner."
            ),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content

    except Exception as e:

        flight_data = f"Flight information unavailable: {str(e)}"

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(
                content="Flight recommendations generated"
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def hotel_agent(state:TravelState):
    query=f"Best hotels for {state['user_query']}"
    # hotel_results=tavily_search(query)
    hotel_results=asyncio.run(tavily_mcp_search(query))

    return {
        "hotel_results":hotel_results,
        "messages":[
            AIMessage(content="Hotel results fetched")
        ],
        "llm_calls":state.get("llm_calls",0)+1
        
    }
    
    
#weather agent 
def weather_agent(state:TravelState):
    city=extract_destination(state["user_query"])
    weather_data=asyncio.run(weather_mcp_seacrh(city))
    forecast_data=asyncio.run(forecast_mcp_search(city))
    return {
        "weather_results": f"""
        Current Weather:
        {weather_data}

        Forecast:
        {forecast_data}
        """,
        "messages": [
            AIMessage(
                content="Weather information fetched"
            )
        ]
    }
#budget agent
def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

#itinerary agent 
def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Budget Results:
{state.get('budget_results', '')}

Make the itinerary practical, budget-aware, and easy to follow.
Create a clear draft that is ready for human review.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
#human in the loop 
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }



#final response agent
def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights:
{state.get('flight_results', '')}

Hotels:
{state.get('hotel_results', '')}

Weather:
{state.get('weather_results', '')}

Budget Analysis:
{state.get('budget_results', '')}

Draft Itinerary:
{state.get('itinerary', '')}

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
- Incorporate the human feedback when revision was requested.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a professional AI travel booking assistant."
            ),
            HumanMessage(content=final_prompt),
        ]
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }
    
    
    
#dynamic supervisor routing 
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}
def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]
def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str): #gives the next agent to execute
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


#build graph 
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)#route_from_supervisor — called with the graph’s current state. It returns a route label, such as "flight_agent" or "guardrail_blocked".ROUTE_MAP — maps that label to the actual graph node to visit. For example, "flight_agent" maps to the "flight_agent" node.
#For example, if the supervisor rejects the request, route_from_supervisor returns "guardrail_blocked". LangGraph looks that up in ROUTE_MAP and routes to the guardrail_blocked node. If the request is allowed, it routes to the first selected agent—or to the itinerary agent if none were selected.
graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)


#postgres checkpointer 

DATABSE_URL=get_database_url()
_conn=psycopg.connect(
    DATABSE_URL,
    autocommit=True,
    row_factory=dict_row#row_factory=dict_row configures Psycopg to return database rows as dictionaries keyed by column name, rather than tuples indexed by position {"id": 1, "name": "Alex"} Then you can access row["name"] instead of row[1]
)
checkpointer=PostgresSaver(_conn)
checkpointer.setup()
travel_graph=graph.compile(checkpointer=checkpointer)



## FastAPI-facing helpers
def _interrupt_payload(result:dict[str,Any])->dict[str,Any] | None:#This helper extracts the human-review prompt from a LangGraph result that paused at an interrupt().
    interrupts=result.get("__interrupt__",[])
    if not interrupts:
        return 
#     {
#     "__interrupt__": (
#         Interrupt(
#             value={
#                 "question": "Do you approve this itinerary?",
#                 "draft_itinerary": "...",
#                 "approval_request": "...",
#                 "selected_agents": [...],
#                 "supervisor_reasoning": "...",
#                 "expected_response": {
#                     "approved": True,
#                     "feedback": "Optional revision feedback"
#                 }
#             },
#             id="..."
#         ),
#     )
# }
    first_interrupt=interrupts[0]#interrupts is the tuple containing one Interrupt object. this  gets the first item from that tuple: the Interrupt object with the value and id fields Gets the object’s .value field. Here, that is the dictionary containing "question", "draft_itinerary", and the other review details.
    payload=getattr(first_interrupt,"value",first_interrupt)#first_interrupt — inspect this object."value" — look for its .value attribute.first_interrupt — if it has no .value attribute, use the object itself. i.e default return
    return payload if isinstance(payload,dict) else {"value":payload}#Checks whether payload is a dictionary. In this example, it is, so the function returns that dictionary directly
def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


#function for fast api

def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)
def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)





#entire flow 
# 1. User requests a trip
# The website sends the request to POST /api/travel. In app.py, that endpoint calls run_travel_agent(...).

# 2. Backend runs the graph and creates a draft
# run_travel_agent(...) starts LangGraph with the user’s request and a thread_id. The graph runs the supervisor and selected agents, then the itinerary agent creates a draft. The PostgreSQL checkpointer saves graph state associated with that thread ID.

# 3. Graph pauses for human review The graph reaches human_approval_agent, which calls:

# review = interrupt({
#     "question": "Do you approve this itinerary?",
#     "draft_itinerary": ...,
#     "approval_request": ...,
# })
# This pauses the graph. At this moment, there is no human feedback yet. The interrupt value contains information the backend wants to show the user: the question and draft.

# 4. Backend prepares and sends the draft
# run_travel_agent(...) receives the paused graph result. _serialize_result(result, thread_id) calls _interrupt_payload(result), which extracts the prompt and draft from the __interrupt__ entry.

# The serializer returns an API response containing the draft, thread_id, and requires_approval: True. It is packaging backend-generated review information, not retrieving written feedback.
# 5. User reviews and responds on the page
# The website displays the draft and feedback box. The user either approves it or types revision feedback and chooses to revise. The page sends that response to POST /api/travel/approve, including the same thread_id.

# 6. Backend resumes the paused graph
# The approval endpoint calls resume_travel_agent(...). That function invokes LangGraph with:
#     Command(resume={
#     "approved": approved,
#     "feedback": feedback,
# })
# Because it uses the same thread_id, LangGraph can find the paused checkpoint. The pending interrupt(...) call now returns this submitted dictionary as review. This is when the human’s response enters the graph.

# 7. Graph creates the final answer
# human_approval_agent reads review["approved"] and review["feedback"], stores them in state, and the graph continues to final_agent. The final agent uses the feedback when producing the final plan. The result is serialized and returned to the page, which displays the final answer.

# In one sentence: _interrupt_payload extracts the draft and question sent to the human; Command(resume=...) carries the human’s approval or feedback back into the paused graph.





#questions

# 1. Does typing feedback automatically set approved=False?
# No. The button the user clicks determines the approved value.

# Approve & Generate Final calls submitApproval(true), so the request sends "approved": true.
# Revise Using Feedback calls submitApproval(false), so it sends "approved": false.
# Typing feedback alone doesn’t change that value. If the user types feedback but clicks Approve, the backend receives approved=True and that feedback; the final agent treats the draft as approved. To request a revision, they need to click Revise Using Feedback. The page also requires feedback when that button is used.

# 2. What does requires_approval mean?
# "requires_approval": interrupt_payload is not None
# It means: did this graph result include a pending interrupt?

# On the first run, the graph pauses at interrupt(...), so _interrupt_payload(result) finds the interrupt and requires_approval is True. The frontend then shows the approval controls.
# After the user responds and the graph finishes, there is no pending interrupt in the result, so it is False. The frontend hides the approval controls.
# It does not mean “the user approved.” The user’s decision is represented separately by approved.

# 3. Why does approval_request have two sources?
# "approval_request": (
#     interrupt_payload.get("approval_request", "")
#     if interrupt_payload
#     else result.get("approval_request", "")
# )
# While paused: use the value inside the interrupt. That is the review message sent to the frontend with the draft.
# When there is no interrupt: use approval_request from the graph’s regular state/result.
# This field is the review instruction or prompt, not the feedback the user typed. The feedback comes back separately through the approval endpoint and is stored as human_feedback.

# 4. Does resuming rerun the whole graph and pause at HITL again?
# No—not from the beginning in the usual flow. The same thread_id lets LangGraph find the checkpoint where the graph paused. Command(resume=...) supplies the user’s response to the pending interrupt(...); then the graph continues from that approval step to final_agent and ends.

# The approval node’s code is revisited to complete the interrupted operation, but the earlier supervisor and specialist agents are not normally rerun from the start. In this graph, the next edge after human_approval leads to final_agent, so the normal resume path does not pause at HITL a second time.

# Flow: first request creates draft → interrupt pauses and returns requires_approval=True → user clicks Approve or Revise → button sends the explicit boolean and feedback → same thread resumes → final agent returns the finished plan



#after human approves graph resumes and  goes to final agent for human approval node and final agent will check response of human and gives the final draft


#in resume graph fucntion also we called serialize again what will happen there?
# The key is that _serialize_result(result, thread_id) is called twice with two different result dictionaries.

# First call: initial run pauses
# The initial travel_graph.invoke(...) pauses at interrupt(...). Its result contains "__interrupt__", so:
#     _interrupt_payload(result)
# finds the interrupt and returns its prompt/draft dictionary. It returns {"value": payload} only if the interrupt’s .value is not already a dictionary. In your code, it is a dictionary, so that wrapper usually isn’t used.

# Second call: resume completes
# resume_travel_agent(...) invokes the graph with Command(resume=...). LangGraph gives that submitted response to the pending interrupt(...) call inside human_approval_agent, where it becomes review. The node reads it, and the graph continues to final_agent.

# The new result from this invocation normally has no pending "__interrupt__" entry. Therefore:

# interrupts = result.get("__interrupt__", [])
# becomes [], and _interrupt_payload(result) returns None.

# So the human response is not returned as _interrupt_payload. It is consumed inside human_approval_agent and saved in the graph state as approved and human_feedback. Serialization then reads those state values and the final response.

# Initial result: contains pending interrupt → _interrupt_payload returns prompt/draft
# Resumed result: interrupt has been answered → _interrupt_payload returns None
# If the resumed graph paused at a new interrupt, then the new result would contain "__interrupt__" and the helper would return that new interrupt’s payload.