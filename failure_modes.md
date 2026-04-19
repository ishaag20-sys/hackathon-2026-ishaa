# Failure Mode Analysis
## ShopWave Autonomous Support Agent

---

### Failure 1: Tool Timeout (`get_customer`)

**What happens:** The `get_customer` tool returns `{"error": "timeout"}` (8% chance per call).

**Agent response:**
1. Detects `error == "timeout"` in the response
2. Logs the failure to `error_log`
3. Waits 500ms (backoff)
4. Retries the call exactly once
5. If retry also fails → proceeds without customer data, confidence drops, agent likely escalates

**Code location:** `node_lookup()` → timeout check + retry block

---

### Failure 2: Malformed Order Data (`get_order`)

**What happens:** The `get_order` tool returns `{"error": "malformed", "raw": "??garbled??"}` (7% chance).

**Agent response:**
1. Detects `error == "malformed"`
2. Logs it to `error_log` as a warning
3. Stores the error state — does NOT crash
4. Downstream nodes skip product lookup and eligibility check
5. LLM is informed of the error and lowers confidence
6. Agent typically escalates with a note about missing order data

**Code location:** `node_lookup()` → malformed check block

---

### Failure 3: Eligibility Service Down (`check_refund_eligibility`)

**What happens:** Returns `{"error": "service_error"}` (10% chance).

**Agent response:**
1. Detects `error == "service_error"`
2. Logs to `error_log`
3. Retries once after 300ms
4. If retry also fails → eligibility remains `{}` (empty)
5. `node_act()` sees no eligibility → does NOT issue refund (safety-first)
6. Escalates to human with note: "eligibility service unavailable"

**Why this matters:** `issue_refund` is irreversible. The agent never refunds without confirmed eligibility.

**Code location:** `node_check_eligibility()` → service_error retry block

---

### Failure 4: LLM Returns Invalid JSON

**What happens:** Gemini occasionally returns markdown-wrapped JSON or malformed JSON.

**Agent response:**
1. `node_decide()` strips markdown fences (` ```json ``` `)
2. Attempts `json.loads()`
3. If still fails → catches the exception
4. Falls back to a safe default: escalate with `confidence=0.3`
5. Logs the raw LLM error string

**Code location:** `node_decide()` → try/except block

---

### Failure 5: Low Confidence Decision

**What happens:** LLM returns confidence < 0.6 for ambiguous tickets (e.g. TKT-020: "my thing is broken").

**Agent response:**
1. `node_act()` checks confidence before any action
2. If `confidence < 0.6` → overrides decision to escalate regardless
3. Sends a polite holding reply to the customer
4. Escalates with the LLM's reasoning attached

**This is a feature:** The agent knows what it doesn't know.

**Code location:** `node_act()` → `if confidence < 0.6` guard
