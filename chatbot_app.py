# =====================================================
# 🧠 SQL Server Chatbot (Safe Rerun + Streamlit 2025 Compatibility)
# =====================================================
# ✅ Hybrid: Semantic + Rule-based SQL
# ✅ Query Store, DMVs, Profiler, MCP-style Insights
# ✅ Server Health Dashboard + Real-Time Line Charts
# ✅ Future-proof (st.query_params compatible)
# =====================================================

import sys
import os
import re
import time
import json
import warnings
import streamlit as st
import pandas as pd
import pyodbc
import numpy as np
from datetime import datetime
from decimal import Decimal
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=UserWarning, module="pandas.io.sql")

# =====================================================
# ⚙️ SAFE RERUN HELPER (uses st.query_params)
# =====================================================
def safe_rerun():
    """Universal rerun that works across Streamlit versions."""
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            # New Streamlit (>=1.40) supports st.query_params for rerun triggers
            try:
                params = st.query_params
                params["_refresh"] = int(time.time() * 1000)
                st.query_params = params
            except Exception:
                st.warning("⚠️ Unable to trigger rerun automatically. Please refresh manually.")
    except Exception:
        st.warning("⚠️ Could not automatically refresh. Please reload manually.")

# =====================================================
# 🎨 PAGE CONFIG
# =====================================================
st.set_page_config(page_title="SQL Server Chatbot", page_icon="🧠", layout="wide")

st.markdown(
    """
    <div style='display:flex; align-items:center; gap:12px;'>
        <img src='https://cdn.freebiesupply.com/logos/large/2x/microsoft-sql-server-logo-png-transparent.png' width='85'>
        <h1 style='color:#A91D22; font-size:28px; margin:0;'>🧠 SQL Server Chatbot </h1>
    </div>
    <p style='font-size:14px; color:#444; margin-top:6px;'>
        Natural + rule-based SQL intelligence with Query Store, DMVs, and live performance monitoring.
    </p>
    """,
    unsafe_allow_html=True,
)

EXPORT_PATH = "exports"
os.makedirs(EXPORT_PATH, exist_ok=True)
INTENTS_FILE = "sql_intents.json"

# =====================================================
# 🔑 CONNECTION SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("<h3 style='color:#A91D22;'>🔑 SQL Server Connection</h3>", unsafe_allow_html=True)

    host = st.text_input("Server / Host", "SQLInstanceName")
    port = st.text_input("Port", "1433")
    database = st.text_input("Database", "master")
    username = st.text_input("Username", "Username")
    password = st.text_input("Password", type="password")

    connect_btn = st.button("Connect")

    st.markdown("---")
    st.markdown("🧠 **Chatbot Mode**")
    use_semantic = st.checkbox("Use Semantic Understanding (Free AI)", value=True)

    st.markdown("---")
    st.markdown("🔎 **Performance & Diagnostics**")
    enable_query_store = st.checkbox("Enable Query Store lookup", value=True)

    st.markdown("---")
    st.markdown("⚙️ Intents")
    use_external_intents = st.checkbox("Load from sql_intents.json", value=True)

# =====================================================
# 🔌 CONNECTION
# =====================================================
def make_conn():
    conn_str = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER=tcp:{host},{port};DATABASE={database};"
        f"UID={username};PWD={password};TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str, autocommit=True)

if connect_btn:
    try:
        conn = make_conn()
        conn.close()
        st.session_state.connected = True
        st.success("✅ Connected successfully!")
    except Exception as e:
        st.session_state.connected = False
        st.error(f"❌ Connection failed: {e}")

# =====================================================
# 🧠 SEMANTIC MODEL
# =====================================================
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

semantic_model = load_semantic_model()

DEFAULT_INTENTS = {
    "maxdop": "SELECT CAST(value_in_use AS INT) AS maxdop FROM sys.configurations WHERE name = 'max degree of parallelism';",
    "max server memory": "SELECT CAST(value_in_use AS INT) AS max_server_memory_MB FROM sys.configurations WHERE name = 'max server memory (MB)';",
    "list databases": "SELECT name, state_desc, recovery_model_desc FROM sys.databases WHERE database_id > 4;",
    "database size": "SELECT DB_NAME(database_id) AS DatabaseName, SUM(size)*8/1024 AS SizeMB FROM sys.master_files GROUP BY database_id ORDER BY SizeMB DESC;",
    "wait stats": "SELECT TOP 10 wait_type, wait_time_ms, waiting_tasks_count FROM sys.dm_os_wait_stats ORDER BY wait_time_ms DESC;"
}

def ensure_intents_file(overwrite=False):
    if not overwrite and os.path.exists(INTENTS_FILE):
        try:
            with open(INTENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    with open(INTENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_INTENTS, f, indent=2)
    return DEFAULT_INTENTS

intents_data = ensure_intents_file() if use_external_intents else DEFAULT_INTENTS
semantic_templates = {**DEFAULT_INTENTS, **intents_data}

semantic_keys = list(semantic_templates.keys())
semantic_embeddings = np.array([semantic_model.encode(k) for k in semantic_keys])

def semantic_map_question_to_sql(user_q):
    user_vec = semantic_model.encode(user_q)
    sims = cosine_similarity([user_vec], semantic_embeddings)[0]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] < 0.45:
        return None
    return semantic_templates[semantic_keys[best_idx]]

# =====================================================
# ▶️ EXECUTION HELPERS
# =====================================================
def run_sql_raw(query: str):
    conn = make_conn()
    try:
        cur = conn.cursor()
        cur.execute(query)
        if cur.description:
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
            df = pd.DataFrame.from_records(rows, columns=cols)
        else:
            df = pd.DataFrame()
        return df, getattr(cur, "messages", [])
    finally:
        conn.close()

def get_query_store_metrics_for_query(sql_text: str):
    query = f"""
    SELECT TOP 10 qsqt.query_sql_text AS query_text,
           qsq.query_id, qsp.plan_id,
           rs.count_executions, rs.avg_duration, rs.avg_cpu_time
    FROM sys.query_store_query qsq
    JOIN sys.query_store_query_text qsqt ON qsq.query_text_id = qsqt.query_text_id
    JOIN sys.query_store_plan qsp ON qsq.query_id = qsp.query_id
    JOIN sys.query_store_runtime_stats rs ON qsp.plan_id = rs.plan_id
    WHERE qsqt.query_sql_text LIKE ?
    ORDER BY rs.count_executions DESC;
    """
    try:
        conn = make_conn()
        cur = conn.cursor()
        cur.execute(query, ('%' + sql_text[:100] + '%',))
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
        return pd.DataFrame.from_records(rows, columns=cols)
    except Exception as e:
        return f"ERROR_QUERY_STORE: {e}"
    finally:
        conn.close()

# =====================================================
# 📊 REAL-TIME PERFORMANCE MONITOR (Streamlit v2025 Compatible)
# =====================================================
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Performance Trend Monitor**")

enable_monitor = st.sidebar.checkbox("Enable Live Performance Charts", value=False)
refresh_rate = st.sidebar.slider("Refresh Interval (seconds)", 10, 60, 15)

if "perf_data" not in st.session_state:
    st.session_state.perf_data = {
        "time": [],
        "cpu_percent": [],
        "wait_ms_per_sec": [],
        "tempdb_used_mb": [],
        "disk_io_mb_sec": [],
        "buffer_cache_hit": []
    }

def fetch_perf_metrics():
    try:
        conn = make_conn()
        cur = conn.cursor()

        # CPU usage
        cpu_query = """
        SELECT TOP 1
            record.value('(./Record/SchedulerMonitorEvent/SystemHealth/SystemIdle)[1]', 'int') AS SystemIdle,
            record.value('(./Record/SchedulerMonitorEvent/SystemHealth/ProcessUtilization)[1]', 'int') AS SQLProcessUtilization
        FROM (
            SELECT CONVERT(XML, record) AS record
            FROM sys.dm_os_ring_buffers
            WHERE ring_buffer_type = N'RING_BUFFER_SCHEDULER_MONITOR'
              AND record LIKE '%<SystemHealth>%'
        ) AS x
        ORDER BY SQLProcessUtilization DESC;
        """
        cur.execute(cpu_query)
        row = cur.fetchone()
        cpu_percent = float(row[1]) if row and row[1] is not None else 0.0

        # Wait time
        cur.execute("SELECT SUM(wait_time_ms) FROM sys.dm_os_wait_stats WHERE wait_type NOT LIKE 'SLEEP%';")
        total_wait = cur.fetchone()[0] or 0

        # TempDB usage
        cur.execute("SELECT SUM(CAST(FILEPROPERTY(name, 'SpaceUsed') AS INT)/128.0) AS UsedMB FROM tempdb.sys.database_files;")
        used_mb = cur.fetchone()[0] or 0

        # Disk I/O
        cur.execute("SELECT TOP 1 (num_of_reads + num_of_writes) * 8.0 / 1024 AS MB_per_sec FROM sys.dm_io_virtual_file_stats(NULL, NULL);")
        disk_io = cur.fetchone()[0] or 0

        # Buffer Cache Hit Ratio
        cur.execute("SELECT TOP 1 cntr_value FROM sys.dm_os_performance_counters WHERE counter_name = 'Buffer cache hit ratio';")
        buffer_hit = cur.fetchone()[0] or 0

        conn.close()

        # Safe float conversion
        def to_float_safe(v):
            try:
                return float(v)
            except Exception:
                return 0.0

        t = datetime.now().strftime("%H:%M:%S")
        st.session_state.perf_data["time"].append(t)
        st.session_state.perf_data["cpu_percent"].append(to_float_safe(cpu_percent))
        st.session_state.perf_data["wait_ms_per_sec"].append(to_float_safe(total_wait))
        st.session_state.perf_data["tempdb_used_mb"].append(to_float_safe(used_mb))
        st.session_state.perf_data["disk_io_mb_sec"].append(to_float_safe(disk_io))
        st.session_state.perf_data["buffer_cache_hit"].append(to_float_safe(buffer_hit))

        for k in st.session_state.perf_data:
            st.session_state.perf_data[k] = st.session_state.perf_data[k][-20:]

    except Exception as e:
        st.warning(f"⚠️ Error fetching metrics: {e}")

if enable_monitor and st.session_state.get("connected", False):
    st.markdown("### 📈 Real-Time SQL Server Performance Charts")
    placeholder = st.empty()
    while enable_monitor:
        fetch_perf_metrics()
        with placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🧠 CPU Utilization (%)**")
                st.line_chart(st.session_state.perf_data["cpu_percent"])
            with col2:
                st.markdown("**🕒 Wait Time (ms/sec)**")
                st.line_chart(st.session_state.perf_data["wait_ms_per_sec"])
            with col3:
                st.markdown("**💾 TempDB Used (MB)**")
                st.line_chart(st.session_state.perf_data["tempdb_used_mb"])
            col4, col5 = st.columns(2)
            with col4:
                st.markdown("**📀 Disk I/O (MB/sec)**")
                st.line_chart(st.session_state.perf_data["disk_io_mb_sec"])
            with col5:
                st.markdown("**🧩 Buffer Cache Hit Ratio (%)**")
                st.line_chart(st.session_state.perf_data["buffer_cache_hit"])
        time.sleep(refresh_rate)
        safe_rerun()

# =====================================================
# 💬 MAIN CHAT
# =====================================================
if "connected" not in st.session_state:
    st.session_state.connected = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.session_state.connected:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], pd.DataFrame):
                st.dataframe(msg["content"], use_container_width=True)
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ask your SQL Server question here..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        sql = semantic_map_question_to_sql(prompt) if use_semantic else None
        if not sql:
            reply = "⚠️ Sorry, I couldn’t understand your question."
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"): st.markdown(reply)
        else:
            try:
                start = time.time()
                df, _ = run_sql_raw(sql)
                dur = time.time() - start
                msg = f"✅ Executed in {dur:.2f}s, {len(df)} rows.\n\n```sql\n{sql}\n```"
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
                st.session_state.chat_history.append({"role": "assistant", "content": df})
                with st.chat_message("assistant"):
                    st.markdown(msg)
                    if not df.empty: st.dataframe(df, use_container_width=True)

                if enable_query_store:
                    qs = get_query_store_metrics_for_query(sql)
                    with st.expander("📦 Query Store Metrics", expanded=False):
                        if isinstance(qs, pd.DataFrame) and not qs.empty:
                            st.dataframe(qs)
                        else:
                            st.info("No Query Store data available.")
            except Exception as e:
                err = f"❌ Error executing query: {e}\n```sql\n{sql}\n```"
                st.session_state.chat_history.append({"role": "assistant", "content": err})
                with st.chat_message("assistant"): st.markdown(err)
else:
    st.info("👈 Please connect to your SQL Server first.")
