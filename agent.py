"""
ShopWave Autonomous Support Resolution Agent
Built with LangGraph + Google Gemini
"""

import json
import os
import random
import time
import asyncio
from datetime import datetime
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────

def load_json(filename):
    with open(f"data/{filename}") as f:
        return json.load(f)

def load_text(filename):
    with open(f"data/{filename}") as f:
        return f.read()

CUSTOMERS = {c["customer_id"]: c for c in load_json("customers.json")}
CUSTOMERS_BY_EMAIL = {c["email"]: c for c in load_json("customers.json")}
ORDERS = {o["order_id"]: o for o in load_json("orders.json")}
PRODUCTS = {p["product_id"]: p for p in load_json("products.json")}
TICKETS = load_json("tickets.json")
KNOWLEDGE_BASE = load_text("knowledge-base.md")


# ─────────────────────────────────────────
# 2. MOCK TOOLS  (some fail realistically)
# ─────────────────────────────────────────

TOOL_CALL_LOG = []   # global audit of every tool call

def _log_tool(ticket_id, tool_name, input_data, output_data):
    TOOL_CALL_LOG.append({
        "ticket_id": ticket_id,
        "tool": tool_name,
        "input": input_data,
        "output": output_data,
        "timestamp": datetime.utcnow().isoformat()
    })

def get_customer(email: str, ticket_id: str = "?") -> dict:
    """Look up a customer by email."""
    # Simulate occasional timeout
    if random.random() < 0.08:
        _log_tool(ticket_id, "get_customer", {"email": email}, {"error": "timeout"})
        return {"error": "timeout", "message": "Service temporarily unavailable"}

    customer = CUSTOMERS_BY_EMAIL.get(email)
    result = customer if customer else {"error": "not_found", "message": f"No customer with email {email}"}
    _log_tool(ticket_id, "get_customer", {"email": email}, result)
    return result

def get_order(order_id: str, ticket_id: str = "?") -> dict:
    """Look up an order by ID."""
    # Simulate occasional malformed response
    if random.random() < 0.07:
        _log_tool(ticket_id, "get_order", {"order_id": order_id}, {"error": "malformed", "raw": "??garbled??"})
        return {"error": "malformed", "order_id": order_id, "message": "Malformed response from order service"}

    order = ORDERS.get(order_id)
    result = dict(order) if order else {"error": "not_found", "order_id": order_id, "message": f"Order {order_id} does not exist"}
    _log_tool(ticket_id, "get_order", {"order_id": order_id}, result)
    return result

def get_product(product_id: str, ticket_id: str = "?") -> dict:
    """Look up product details."""
    product = PRODUCTS.get(product_id)
    result = product if product else {"error": "not_found", "message": f"Product {product_id} not found"}
    _log_tool(ticket_id, "get_product", {"product_id": product_id}, result)
    return result

def check_refund_eligibility(order_id: str, ticket_id: str = "?") -> dict:
    """Check if an order is eligible for a refund. May throw errors."""
    # Simulate errors
    if random.random() < 0.1:
        _log_tool(ticket_id, "check_refund_eligibility", {"order_id": order_id}, {"error": "service_error"})
        return {"error": "service_error", "message": "Eligibility service is down. Retry later."}

    order = ORDERS.get(order_id)
    if not order:
        result = {"eligible": False, "reason": f"Order {order_id} not found"}
        _log_tool(ticket_id, "check_refund_eligibility", {"order_id": order_id}, result)
        return result

    # Already refunded?
    if order.get("refund_status") == "refunded":
        result = {"eligible": False, "reason": "Refund already processed for this order"}
        _log_tool(ticket_id, "check_refund_eligibility", {"order_id": order_id}, result)
        return result

    # Order not delivered yet
    if order["status"] in ["processing", "shipped"]:
        result = {"eligible": False, "reason": f"Order is still {order['status']} — cannot refund yet"}
        _log_tool(ticket_id, "check_refund_eligibility", {"order_id": order_id}, result)
        return result

    # Check return deadline
    if order.get("return_deadline"):
        deadline = datetime.strptime(order["return_deadline"], "%Y-%m-%d")
        today = datetime(2024, 3, 22)   # frozen to hackathon dataset date
        if today > deadline:
            # Check customer tier for leniency
            customer_id = order.get("customer_id")
            customer = CUSTOMERS.get(customer_id, {})
            tier = customer.get("tier", "standard")
            notes = customer.get("notes", "")
            if tier == "vip" and "pre-approved" in notes.lower():
                result = {"eligible": True, "reason": "VIP customer with management pre-approval on file"}
                _log_tool(ticket_id, "check_refund_eligibility", {"order_id": order_id}, result)
                return result
            result = {"eligible": False, "reason": f"Return window expired on {order['return_deadline']}"}
            _log_tool(ticket_id, "check_refund_eligibility", {"order_id": order_id}, result)
            return result

    result = {"eligible": True, "reason": "Order is within return window and meets refund criteria"}
    _log_tool(ticket_id, "check_refund_eligibility", {"order_id": order_id}, result)
    return result

def search_knowledge_base(query: str, ticket_id: str = "?") -> str:
    """Search the ShopWave policy knowledge base."""
    # Simple keyword search — returns the most relevant section
    kb = KNOWLEDGE_BASE.lower()
    query_lower = query.lower()
    keywords = query_lower.split()

    # Find the best matching section
    sections = KNOWLEDGE_BASE.split("\n## ")
    best_section = ""
    best_score = 0
    for section in sections:
        score = sum(1 for kw in keywords if kw in section.lower())
        if score > best_score:
            best_score = score
            best_section = section

    result = best_section[:1200] if best_section else "No relevant policy found."
    _log_tool(ticket_id, "search_knowledge_base", {"query": query}, {"result_preview": result[:100] + "..."})
    return result

def issue_refund(order_id: str, amount: float, ticket_id: str = "?") -> dict:
    """IRREVERSIBLE — issue a refund. Must check eligibility first."""
    eligibility = check_refund_eligibility(order_id, ticket_id)
    if eligibility.get("error"):
        result = {"success": False, "reason": f"Could not verify eligibility: {eligibility['message']}"}
        _log_tool(ticket_id, "issue_refund", {"order_id": order_id, "amount": amount}, result)
        return result
    if not eligibility.get("eligible"):
        result = {"success": False, "reason": eligibility.get("reason", "Not eligible")}
        _log_tool(ticket_id, "issue_refund", {"order_id": order_id, "amount": amount}, result)
        return result

    result = {"success": True, "refund_id": f"REF-{order_id}-{int(time.time())}", "amount": amount, "message": f"Refund of ${amount} initiated. 5–7 business days to process."}
    _log_tool(ticket_id, "issue_refund", {"order_id": order_id, "amount": amount}, result)
    return result

def send_reply(ticket_id: str, message: str) -> dict:
    """Send a reply to the customer."""
    result = {"success": True, "ticket_id": ticket_id, "message_sent": message[:80] + "..."}
    _log_tool(ticket_id, "send_reply", {"ticket_id": ticket_id, "message_preview": message[:80]}, result)
    return result

def escalate(ticket_id: str, summary: str, priority: str) -> dict:
    """Escalate to a human agent with full context."""
    result = {"success": True, "ticket_id": ticket_id, "priority": priority, "assigned_to": "human_support_team"}
    _log_tool(ticket_id, "escalate", {"ticket_id": ticket_id, "summary": summary, "priority": priority}, result)
    return result


# ─────────────────────────────────────────
# 3. AGENT STATE
# ─────────────────────────────────────────

class TicketState(TypedDict):
    ticket: dict
    customer: dict
    order: dict
    product: dict
    eligibility: dict
    kb_result: str
    resolution: str
    reply_message: str
    action_taken: str        # "resolved", "escalated", "replied", "cancelled"
    confidence: float
    escalation_reason: str
    tool_calls_made: list
    error_log: list


# ─────────────────────────────────────────
# 4. LLM SETUP
# ─────────────────────────────────────────

def get_llm():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )


# ─────────────────────────────────────────
# 5. LANGGRAPH NODES
# ─────────────────────────────────────────

def node_lookup(state: TicketState) -> TicketState:
    """Step 1: Look up customer and order from the ticket."""
    ticket = state["ticket"]
    tid = ticket["ticket_id"]
    errors = state.get("error_log", [])
    calls = state.get("tool_calls_made", [])

    # Get customer
    customer = get_customer(ticket["customer_email"], tid)
    calls.append("get_customer")
    if customer.get("error") == "timeout":
        errors.append("get_customer timed out — retrying once")
        time.sleep(0.5)
        customer = get_customer(ticket["customer_email"], tid)
        calls.append("get_customer(retry)")

    # Extract order ID from ticket body
    order_id = None
    body = ticket.get("body", "") + " " + ticket.get("subject", "")
    import re
    match = re.search(r"ORD-\d+", body)
    if match:
        order_id = match.group(0)

    order = {}
    if order_id:
        order = get_order(order_id, tid)
        calls.append("get_order")
        if order.get("error") == "malformed":
            errors.append(f"get_order returned malformed data for {order_id} — skipping order lookup")
            order = {"error": "malformed", "order_id": order_id}

    return {**state, "customer": customer, "order": order,
            "tool_calls_made": calls, "error_log": errors}


def node_enrich(state: TicketState) -> TicketState:
    """Step 2: Get product details and search knowledge base."""
    ticket = state["ticket"]
    tid = ticket["ticket_id"]
    order = state.get("order", {})
    calls = state.get("tool_calls_made", [])

    # Get product if we have an order
    product = {}
    if order and not order.get("error") and order.get("product_id"):
        product = get_product(order["product_id"], tid)
        calls.append("get_product")

    # Search knowledge base based on ticket subject
    query = ticket.get("subject", "") + " " + ticket.get("body", "")[:100]
    kb_result = search_knowledge_base(query, tid)
    calls.append("search_knowledge_base")

    return {**state, "product": product, "kb_result": kb_result, "tool_calls_made": calls}


def node_check_eligibility(state: TicketState) -> TicketState:
    """Step 3: Check refund eligibility if relevant."""
    ticket = state["ticket"]
    tid = ticket["ticket_id"]
    order = state.get("order", {})
    calls = state.get("tool_calls_made", [])
    errors = state.get("error_log", [])

    eligibility = {}
    body_lower = (ticket.get("body", "") + ticket.get("subject", "")).lower()
    refund_keywords = ["refund", "return", "cancel", "broken", "damaged", "defect", "wrong"]

    if any(kw in body_lower for kw in refund_keywords) and order and not order.get("error") and order.get("order_id"):
        eligibility = check_refund_eligibility(order["order_id"], tid)
        calls.append("check_refund_eligibility")
        # Handle service error with retry
        if eligibility.get("error") == "service_error":
            errors.append("Eligibility service error — retrying once")
            time.sleep(0.3)
            eligibility = check_refund_eligibility(order["order_id"], tid)
            calls.append("check_refund_eligibility(retry)")

    return {**state, "eligibility": eligibility, "tool_calls_made": calls, "error_log": errors}


def node_decide(state: TicketState) -> TicketState:
    """Step 4: Use Gemini to reason and decide what action to take."""
    llm = get_llm()
    ticket = state["ticket"]
    tid = ticket["ticket_id"]

    system_prompt = f"""You are ShopWave's autonomous support agent. 
You MUST respond with ONLY valid JSON — no markdown, no explanation, just JSON.

KNOWLEDGE BASE:
{state.get('kb_result', '')}

Respond with this exact JSON structure:
{{
  "action": "resolve" or "escalate" or "info_needed",
  "resolution_type": "refund" or "cancel" or "reply_info" or "exchange" or "confirm_status" or "escalate" or "clarify",
  "reply_to_customer": "The actual reply message to send to the customer (friendly, uses their first name)",
  "escalation_summary": "Only if escalating — brief summary for human agent",
  "escalation_priority": "low" or "medium" or "high" or "urgent",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief internal reasoning"
}}"""

    user_msg = f"""TICKET:
ID: {ticket['ticket_id']}
From: {ticket['customer_email']}
Subject: {ticket['subject']}
Body: {ticket['body']}

CUSTOMER DATA: {json.dumps(state.get('customer', {}), indent=2)}
ORDER DATA: {json.dumps(state.get('order', {}), indent=2)}
PRODUCT DATA: {json.dumps(state.get('product', {}), indent=2)}
REFUND ELIGIBILITY: {json.dumps(state.get('eligibility', {}), indent=2)}
ERRORS ENCOUNTERED: {state.get('error_log', [])}

Decide what to do. Reply with ONLY JSON."""

    try:
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
        raw = response.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        decision = json.loads(raw)
    except Exception as e:
        decision = {
            "action": "escalate",
            "resolution_type": "escalate",
            "reply_to_customer": "We are reviewing your request and will get back to you shortly.",
            "escalation_summary": f"LLM decision failed: {str(e)}",
            "escalation_priority": "medium",
            "confidence": 0.3,
            "reasoning": f"LLM error: {str(e)}"
        }

    return {**state,
            "resolution": decision.get("reasoning", ""),
            "reply_message": decision.get("reply_to_customer", ""),
            "action_taken": decision.get("action", "escalate"),
            "confidence": decision.get("confidence", 0.5),
            "escalation_reason": decision.get("escalation_summary", ""),
            "_decision": decision}


def node_act(state: TicketState) -> TicketState:
    """Step 5: Execute the decided action."""
    ticket = state["ticket"]
    tid = ticket["ticket_id"]
    order = state.get("order", {})
    decision = state.get("_decision", {})
    calls = state.get("tool_calls_made", [])
    confidence = state.get("confidence", 0.5)

    # Low confidence → always escalate
    if confidence < 0.6:
        state = {**state, "action_taken": "escalated"}
        escalate(tid, f"Low confidence ({confidence}). " + state.get("resolution", ""), "medium")
        calls.append("escalate")
        send_reply(tid, state.get("reply_message", "We are reviewing your case."))
        calls.append("send_reply")
        return {**state, "tool_calls_made": calls}

    action = decision.get("action", "escalate")
    res_type = decision.get("resolution_type", "reply_info")

    if action == "resolve":
        # Issue refund if eligible
        if res_type == "refund" and order and order.get("order_id") and not order.get("error"):
            eligibility = state.get("eligibility", {})
            if eligibility.get("eligible"):
                amount = order.get("amount", 0)
                issue_refund(order["order_id"], amount, tid)
                calls.append("issue_refund")

        # Cancel order if in processing
        elif res_type == "cancel" and order.get("status") == "processing":
            # Mock cancel — log it
            _log_tool(tid, "cancel_order", {"order_id": order.get("order_id")}, {"success": True})
            calls.append("cancel_order")

        send_reply(tid, state.get("reply_message", "Your request has been processed."))
        calls.append("send_reply")
        return {**state, "action_taken": "resolved", "tool_calls_made": calls}

    elif action == "escalate":
        priority = decision.get("escalation_priority", "medium")
        escalate(tid, state.get("escalation_reason", "Needs human review"), priority)
        calls.append("escalate")
        send_reply(tid, state.get("reply_message", "Your case is being reviewed by our team."))
        calls.append("send_reply")
        return {**state, "action_taken": "escalated", "tool_calls_made": calls}

    else:  # info_needed
        send_reply(tid, state.get("reply_message", "Could you please provide more details?"))
        calls.append("send_reply")
        return {**state, "action_taken": "replied", "tool_calls_made": calls}


# ─────────────────────────────────────────
# 6. BUILD THE GRAPH
# ─────────────────────────────────────────

def build_graph():
    graph = StateGraph(TicketState)

    graph.add_node("lookup", node_lookup)
    graph.add_node("enrich", node_enrich)
    graph.add_node("check_eligibility", node_check_eligibility)
    graph.add_node("decide", node_decide)
    graph.add_node("act", node_act)

    graph.set_entry_point("lookup")
    graph.add_edge("lookup", "enrich")
    graph.add_edge("enrich", "check_eligibility")
    graph.add_edge("check_eligibility", "decide")
    graph.add_edge("decide", "act")
    graph.add_edge("act", END)

    return graph.compile()


# ─────────────────────────────────────────
# 7. PROCESS ALL TICKETS CONCURRENTLY
# ─────────────────────────────────────────

def process_ticket(ticket: dict, agent) -> dict:
    """Process a single ticket through the agent graph."""
    initial_state = TicketState(
        ticket=ticket,
        customer={},
        order={},
        product={},
        eligibility={},
        kb_result="",
        resolution="",
        reply_message="",
        action_taken="",
        confidence=0.0,
        escalation_reason="",
        tool_calls_made=[],
        error_log=[]
    )
    result = agent.invoke(initial_state)
    return result


async def process_all_tickets_async(tickets, agent):
    """Process all tickets concurrently using asyncio."""
    loop = asyncio.get_event_loop()

    tasks = [
        loop.run_in_executor(None, process_ticket, ticket, agent)
        for ticket in tickets
    ]

    results = await asyncio.gather(*tasks)
    return list(results)


# ─────────────────────────────────────────
# 8. AUDIT LOG
# ─────────────────────────────────────────

def save_audit_log(results: list):
    audit = []
    for r in results:
        ticket = r.get("ticket", {})
        audit.append({
            "ticket_id": ticket.get("ticket_id"),
            "customer_email": ticket.get("customer_email"),
            "subject": ticket.get("subject"),
            "action_taken": r.get("action_taken"),
            "confidence": r.get("confidence"),
            "resolution_reasoning": r.get("resolution"),
            "reply_sent": r.get("reply_message"),
            "escalation_reason": r.get("escalation_reason"),
            "tool_calls_made": r.get("tool_calls_made"),
            "errors_encountered": r.get("error_log"),
            "tool_call_details": [
                t for t in TOOL_CALL_LOG
                if t["ticket_id"] == ticket.get("ticket_id")
            ]
        })

    with open("audit_log.json", "w") as f:
        json.dump(audit, f, indent=2)

    print(f"\n✅ audit_log.json saved with {len(audit)} ticket records.")
    return audit


# ─────────────────────────────────────────
# 9. MAIN ENTRY POINT
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ShopWave Autonomous Support Agent")
    print("  Powered by LangGraph + Google Gemini")
    print("=" * 60)

    if not os.environ.get("GEMINI_API_KEY"):
        print("\n⚠️  GEMINI_API_KEY not set!")
        print("  Run: export GEMINI_API_KEY='your-key-here'")
        return

    print(f"\n📋 Loading {len(TICKETS)} tickets...")
    agent = build_graph()

    print("🚀 Processing all tickets concurrently...\n")
    start = time.time()

    results = asyncio.run(process_all_tickets_async(TICKETS, agent))

    elapsed = time.time() - start

    # Print summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    resolved = sum(1 for r in results if r.get("action_taken") == "resolved")
    escalated = sum(1 for r in results if r.get("action_taken") == "escalated")
    replied = sum(1 for r in results if r.get("action_taken") == "replied")

    for r in results:
        ticket = r.get("ticket", {})
        action = r.get("action_taken", "unknown")
        confidence = r.get("confidence", 0)
        icon = "✅" if action == "resolved" else "🔺" if action == "escalated" else "💬"
        print(f"  {icon} {ticket.get('ticket_id')} | {action:10} | confidence: {confidence:.2f} | {ticket.get('subject', '')[:40]}")

    print(f"\n  Total: {len(results)} tickets")
    print(f"  ✅ Resolved: {resolved}")
    print(f"  🔺 Escalated: {escalated}")
    print(f"  💬 Replied: {replied}")
    print(f"  ⏱️  Time: {elapsed:.1f}s (concurrent)")

    save_audit_log(results)
    print("\n🎉 Done! Check audit_log.json for full details.\n")


if __name__ == "__main__":
    main()
