# ShopWave Autonomous Support Agent
### Ksolves Agentic AI Hackathon 2026

An autonomous AI agent that resolves ShopWave customer support tickets using **LangGraph** + **Google Gemini**, with concurrent processing, intelligent escalation, and full audit logging.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph) |
| LLM | Google Gemini 1.5 Flash |
| Language | Python 3.10+ |
| Concurrency | asyncio + ThreadPoolExecutor |

---

## Setup (3 steps)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key
Get a free key at: https://aistudio.google.com/app/apikey
```bash
export GEMINI_API_KEY="your-key-here"
```

### 3. Run the agent
```bash
python agent.py
```

That's it. The agent will process all 20 tickets concurrently and save `audit_log.json`.

---

## Agent Architecture

```
Ticket Input
    │
    ▼
[lookup]          → get_customer() + get_order()
    │
    ▼
[enrich]          → get_product() + search_knowledge_base()
    │
    ▼
[check_eligibility] → check_refund_eligibility()
    │
    ▼
[decide]          → Gemini LLM reasons and picks action
    │
    ▼
[act]             → issue_refund() / cancel_order() / send_reply() / escalate()
    │
    ▼
audit_log.json
```

All 20 tickets are processed **concurrently** (not one-by-one).

---

## Tools Implemented

| Tool | Type | Description |
|---|---|---|
| `get_customer(email)` | READ | Customer profile, tier, history |
| `get_order(order_id)` | READ | Order status, dates, amounts |
| `get_product(product_id)` | READ | Category, warranty, return window |
| `check_refund_eligibility(order_id)` | READ | Eligibility + reason. May throw errors |
| `search_knowledge_base(query)` | READ | Policy & FAQ search |
| `issue_refund(order_id, amount)` | WRITE | Irreversible — checks eligibility first |
| `send_reply(ticket_id, message)` | WRITE | Sends reply to customer |
| `escalate(ticket_id, summary, priority)` | WRITE | Routes to human with full context |

---

## Failure Handling

| Failure | How Agent Handles It |
|---|---|
| `get_customer` timeout | Retries once with 0.5s backoff |
| `get_order` malformed data | Logs error, continues without order data |
| `check_refund_eligibility` service error | Retries once, then escalates if still failing |
| LLM JSON parse error | Falls back to escalation with error note |
| Confidence < 0.6 | Always escalates to human regardless of action |

---

## Escalation Logic

The agent escalates when:
- Confidence score < 0.6
- Refund amount > $200 (per policy)
- Warranty claim detected (goes to warranty team)
- Social engineering / fraud signals detected
- Customer requests replacement (not refund)
- Tool errors prevent reliable resolution

---

## Output Files

- **`audit_log.json`** — Full record of every ticket: tool calls, reasoning, decisions, errors
