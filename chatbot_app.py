# =====================================================
# 🧠 SQL Server Chatbot (v11 Persistent Chat)
# =====================================================
# ✅ Keeps full chat history (tables + answers)
# ✅ Clears only on browser refresh
# ✅ Works with Python 3.13 + Streamlit
# ✅ Stable PNG SQL Server logo
# =====================================================

import sys
import os
import re
import warnings
import streamlit as st
import pandas as pd
import pyodbc
from datetime import datetime
from decimal import Decimal
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="pandas.io.sql")

# =====================================================
# 🎨 PAGE CONFIG
# =====================================================
st.set_page_config(page_title="SQL Server Chatbot", page_icon="🧠", layout="wide")

st.markdown(
    """
    <div style='display:flex; align-items:center; gap:12px;'>
        <img src='https://cdn.freebiesupply.com/logos/large/2x/microsoft-sql-server-logo-png-transparent.png' width='85'>
        <h1 style='color:#A91D22; font-size:28px; margin:0;'>🧠 SQL Server Chatbot</h1>
    </div>
    <p style='font-size:16px; color:#444; margin-top:8px;'>
        Ask SQL Server Related Questions — Server Level and Instance Level
    </p>
    """,
    unsafe_allow_html=True,
)

EXPORT_PATH = "exports"
os.makedirs(EXPORT_PATH, exist_ok=True)

# =====================================================
# 🔑 CONNECTION SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown(
        "<h3 style='color:#A91D22;'>🔑 SQL Server Connection</h3>",
        unsafe_allow_html=True
    )

    host = st.text_input("Server / Host", "SQLInstanceName")
    port = st.text_input("Port", "1433")
    database = st.text_input("Database", "master")
    username = st.text_input("Username", "Username")
    password = st.text_input("Password", type="password")

    connect_btn = st.button("Connect")

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
# 🧠 MAP QUESTION TO SQL
# =====================================================
def map_question_to_sql(q: str):
    q_clean = q.lower().strip()
    q_parts = q.split()
    q_lower_parts = [p.lower() for p in q_parts]

    # ✅ Download/export
    if "download" in q_clean and "table" in q_clean and "database" in q_clean:
        try:
            years, start_date, end_date = None, None, None
            if "year" in q_clean:
                nums = [int(p) for p in q_parts if p.isdigit()]
                if nums: years = nums[0]
            dates = re.findall(r"\d{4}-\d{2}-\d{2}", q)
            if len(dates) == 2: start_date, end_date = dates
            if "database" in q_lower_parts and "table" in q_lower_parts:
                db = q_parts[q_lower_parts.index("database")+1]
                tbl = q_parts[q_lower_parts.index("table")+1]
                date_col = detect_date_column(db, tbl)
                if years:
                    sql = f"SELECT * FROM [{db}].dbo.[{tbl}] WHERE [{date_col}] >= DATEADD(YEAR, -{years}, GETDATE());"
                elif start_date and end_date:
                    sql = f"SELECT * FROM [{db}].dbo.[{tbl}] WHERE [{date_col}] >= '{start_date}' AND [{date_col}] <= '{end_date}';"
                else:
                    sql = f"SELECT * FROM [{db}].dbo.[{tbl}];"
                return {"export": True, "db": db, "tbl": tbl, "sql": sql,
                        "years": years, "start": start_date, "end": end_date, "date_col": date_col}
        except Exception as e:
            return f"❌ Error in auto-detecting date column: {e}"

    # ======================================================
    # Instance-level configs
    # ======================================================
    if "maxdop" in q_clean or "max dop" in q_clean or "max degree" in q_clean:
        return "SELECT CAST(value_in_use AS INT) AS maxdop FROM sys.configurations WHERE name = 'max degree of parallelism';"
    if "max server memory" in q_clean:
        return "SELECT CAST(value_in_use AS INT) AS max_server_memory_MB FROM sys.configurations WHERE name = 'max server memory (MB)';"
    if "min server memory" in q_clean:
        return "SELECT CAST(value_in_use AS INT) AS min_server_memory_MB FROM sys.configurations WHERE name = 'min server memory (MB)';"
    if "cost threshold" in q_clean:
        return "SELECT CAST(value_in_use AS INT) AS cost_threshold FROM sys.configurations WHERE name = 'cost threshold for parallelism';"
    if "service account" in q_clean:
        return "SELECT servicename, service_account FROM sys.dm_server_services;"
    if "version" in q_clean:
        return "SELECT @@VERSION AS SQLServerVersion;"

    # ======================================================
    # Database-level queries
    # ======================================================
    if "list databases" in q_clean or "user databases" in q_clean:
        return """
        SELECT name, state_desc, recovery_model_desc, compatibility_level
        FROM sys.databases
        WHERE database_id > 4
        ORDER BY name;
        """
    if "compatibility level" in q_clean and "database" in q_clean:
        db = q_parts[-1].strip()
        return f"SELECT compatibility_level FROM sys.databases WHERE name = '{db}';"
    if "collation" in q_clean and "database" in q_clean:
        db = q_parts[-1].strip()
        return f"SELECT collation_name FROM sys.databases WHERE name = '{db}';"
    if "server collation" in q_clean:
        return "SELECT SERVERPROPERTY('Collation') AS ServerCollation;"
    if "recovery model" in q_clean and "database" in q_clean:
        db = q_parts[-1].strip()
        return f"SELECT name, recovery_model_desc FROM sys.databases WHERE name = '{db}';"
    if "size of database" in q_clean or "database size" in q_clean:
        db = q_parts[-1].strip()
        return f"""
        SELECT DB_NAME(database_id) AS DatabaseName,
               CAST(SUM(size) * 8.0 / 1024 AS DECIMAL(18,2)) AS SizeMB
        FROM sys.master_files
        WHERE DB_NAME(database_id) = '{db}'
        GROUP BY database_id;
        """
    if "database growth" in q_clean:
        return """
        SELECT d.name AS DatabaseName, mf.name AS FileName,
               mf.size/128 AS CurrentSizeMB,
               mf.max_size/128 AS MaxSizeMB,
               mf.growth/128 AS GrowthMB
        FROM sys.master_files mf
        JOIN sys.databases d ON mf.database_id = d.database_id;
        """
    if "filegroups" in q_clean:
        return "SELECT DB_NAME(database_id) AS DatabaseName, name AS FileGroupName, type_desc, size*8/1024 AS SizeMB FROM sys.master_files;"
    if "tempdb" in q_clean:
        return "SELECT name AS FileName, physical_name, size*8/1024 AS SizeMB, max_size, growth FROM tempdb.sys.database_files;"
    if "dbcc" in q_clean:
        return "DBCC CHECKDB WITH NO_INFOMSGS, ALL_ERRORMSGS;"
    if "free space" in q_clean and "database" in q_clean:
        db = q_parts[-1].strip()
        return f"""
        SELECT DB_NAME(database_id) AS DatabaseName, name AS FileName,
               size/128.0 AS TotalSizeMB,
               size/128.0 - CAST(FILEPROPERTY(name, 'SpaceUsed') AS int)/128.0 AS FreeSpaceMB
        FROM sys.master_files
        WHERE DB_NAME(database_id) = '{db}';
        """

    # ======================================================
    # Tables / Views / Procedures / Functions
    # ======================================================
    if "number of tables" in q_clean or "count of tables" in q_clean:
        db = q_parts[-1].strip()
        return f"SELECT COUNT(*) AS TableCount FROM [{db}].sys.tables;"
    if "list tables" in q_clean:
        db = q_parts[-1].strip()
        return f"SELECT name AS TableName FROM [{db}].sys.tables ORDER BY name;"
    if "list views" in q_clean:
        db = q_parts[-1].strip()
        return f"SELECT name AS ViewName FROM [{db}].sys.views ORDER BY name;"
    if "list procedures" in q_clean:
        db = q_parts[-1].strip()
        return f"SELECT name AS ProcedureName FROM [{db}].sys.procedures ORDER BY name;"
    if "list functions" in q_clean:
        db = q_parts[-1].strip()
        return f"SELECT name AS FunctionName FROM [{db}].sys.objects WHERE type IN ('FN','IF','TF','FS','FT') ORDER BY name;"
    if ("how many rows" in q_clean or "row count" in q_clean) and "table" in q_clean:
        tbl = q_parts[q_parts.index("table")+1]
        db = q_parts[q_parts.index("database")+1]
        return f"SELECT COUNT_BIG(*) AS TotalRows FROM [{db}].dbo.[{tbl}];"
    if "top table" in q_clean:
        db = q_parts[-1].strip()
        return f"""
        SELECT TOP 1 t.name AS TableName, SUM(p.rows) AS RowCounts
        FROM [{db}].sys.tables t
        INNER JOIN [{db}].sys.partitions p ON t.object_id = p.object_id
        WHERE p.index_id IN (0,1)
        GROUP BY t.name
        ORDER BY RowCounts DESC;
        """

    # ======================================================
    # Sessions, Blocking, Deadlocks
    # ======================================================
    if "sessions" in q_clean:
        return "SELECT session_id, login_name, host_name, status, database_id FROM sys.dm_exec_sessions WHERE is_user_process = 1;"
    if "running queries" in q_clean:
        return """
        SELECT r.session_id, r.status, r.command, s.login_name, s.host_name,
               DB_NAME(r.database_id) AS DatabaseName, st.text AS QueryText
        FROM sys.dm_exec_requests r
        JOIN sys.dm_exec_sessions s ON r.session_id = s.session_id
        CROSS APPLY sys.dm_exec_sql_text(r.sql_handle) st
        WHERE r.session_id <> @@SPID;
        """
    if "blocking" in q_clean:
        return "SELECT blocking_session_id, session_id, wait_type, wait_time, wait_resource, text FROM sys.dm_exec_requests CROSS APPLY sys.dm_exec_sql_text(sql_handle) WHERE blocking_session_id <> 0;"
    if "deadlock" in q_clean:
        return """
        SELECT XEvent.query('(event/data/value/deadlock)[1]') AS DeadlockGraph
        FROM (SELECT CAST(target_data AS XML) AS TargetData
              FROM sys.dm_xe_session_targets st
              JOIN sys.dm_xe_sessions s ON s.address = st.event_session_address
              WHERE s.name = 'system_health') AS Data
        CROSS APPLY TargetData.nodes('//RingBufferTarget/event') AS XEventData(XEvent)
        WHERE XEvent.value('@name', 'varchar(100)') = 'xml_deadlock_report';
        """

    # ======================================================
    # Performance tuning
    # ======================================================
    if "fragmentation" in q_clean and "index" in q_clean:
        db = q_parts[q_parts.index("database")+1]
        return f"""
        SELECT OBJECT_NAME(ips.object_id) AS TableName, i.name AS IndexName,
               ips.avg_fragmentation_in_percent, ips.page_count
        FROM sys.dm_db_index_physical_stats(DB_ID('{db}'), NULL, NULL, NULL, 'LIMITED') ips
        JOIN sys.indexes i ON ips.object_id = i.object_id AND ips.index_id = i.index_id
        WHERE ips.database_id = DB_ID('{db}') AND ips.page_count > 1000
        ORDER BY ips.avg_fragmentation_in_percent DESC;
        """
    if "top queries" in q_clean or "high cpu" in q_clean:
        return """
        SELECT TOP 10 qs.total_worker_time/1000 AS CPU_ms, qs.execution_count,
               qs.total_elapsed_time/1000 AS Elapsed_ms,
               SUBSTRING(st.text, (qs.statement_start_offset/2)+1,
               ((CASE qs.statement_end_offset WHEN -1 THEN DATALENGTH(st.text)
                 ELSE qs.statement_end_offset END - qs.statement_start_offset)/2)+1) AS query_text
        FROM sys.dm_exec_query_stats qs
        CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) st
        ORDER BY qs.total_worker_time DESC;
        """
    if "wait stats" in q_clean or "top waits" in q_clean:
        return "SELECT TOP 20 wait_type, wait_time_ms, waiting_tasks_count FROM sys.dm_os_wait_stats ORDER BY wait_time_ms DESC;"

    # ======================================================
    # Security / Users
    # ======================================================
    if "how many users" in q_clean and "database" in q_clean:
        db = q_parts[-1].strip()
        return f"SELECT COUNT(*) AS UserCount FROM [{db}].sys.database_principals WHERE type_desc IN ('SQL_USER','WINDOWS_USER','DATABASE_ROLE') AND principal_id > 4;"
    if "orphan users" in q_clean:
        return "SELECT dp.name AS OrphanUser FROM sys.database_principals dp LEFT JOIN sys.server_principals sp ON dp.sid = sp.sid WHERE sp.sid IS NULL AND dp.type_desc = 'SQL_USER';"
    if "password" in q_clean and "policy" in q_clean:
        return "SELECT name, is_expiration_checked, is_policy_checked FROM sys.sql_logins;"

    # ======================================================
    # Agent Jobs / Monitoring
    # ======================================================
    if "how many jobs" in q_clean or "list jobs" in q_clean:
        return "SELECT COUNT(*) AS JobCount FROM msdb.dbo.sysjobs;"
    if "maintenance plan" in q_clean:
        return "SELECT name, date_modified FROM msdb.dbo.sysmaintplan_plans;"
    if "linked servers" in q_clean:
        return "SELECT name, data_source, provider, catalog FROM sys.servers WHERE is_linked = 1;"
    if "extended events" in q_clean:
        return "SELECT name, event_session_id, startup_state FROM sys.server_event_sessions;"
    if "database mail" in q_clean:
        return "SELECT * FROM msdb.dbo.sysmail_profile;"
    if "last backup" in q_clean and "database" in q_clean:
        db = q_parts[-1].strip()
        return f"""
        SELECT TOP 1 d.name AS DatabaseName, b.backup_finish_date AS LastBackupDate,
               CASE b.type WHEN 'D' THEN 'Full' WHEN 'I' THEN 'Differential' WHEN 'L' THEN 'Log' ELSE b.type END AS BackupType
        FROM msdb.dbo.backupset b
        INNER JOIN sys.databases d ON b.database_name = d.name
        WHERE d.name = '{db}'
        ORDER BY b.backup_finish_date DESC;
        """
        # ======================================================
    # Server Connections and Jobs
    # ======================================================
    if "connections" in q_clean or "current connections" in q_clean or "active connections" in q_clean:
        return "SELECT COUNT(*) AS CurrentConnections FROM sys.dm_exec_connections;"

    if "running jobs" in q_clean or "current running jobs" in q_clean or "sql agent running jobs" in q_clean:
        return """
        SELECT job.name AS JobName, ja.start_execution_date, ja.stop_execution_date, ja.job_id
        FROM msdb.dbo.sysjobactivity ja
        JOIN msdb.dbo.sysjobs job ON ja.job_id = job.job_id
        WHERE ja.stop_execution_date IS NULL
        AND ja.start_execution_date IS NOT NULL;
        """


    # ======================================================
    # Direct SQL fallback
    # ======================================================
    if q_clean.startswith(("select", "exec", "with")):
        return q

    return None


# =====================================================
# ▶️ RUN SQL
# =====================================================
from decimal import Decimal
import pandas as pd
import numpy as np

def run_sql(query: str):
    conn = make_conn()
    try:
        cur = conn.cursor()
        cur.execute(query)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()

        df = pd.DataFrame.from_records(rows, columns=cols)

        # ✅ Convert Decimal or string numbers → float
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: float(x)
                if isinstance(x, (Decimal, int, float, str)) and str(x).replace('.', '', 1).isdigit()
                else x
            )

        # ✅ Convert any non-numeric to NaN to allow rounding
        for col in df.select_dtypes(include="object").columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        # ✅ Add GB columns if MB columns exist
        if "TotalSizeMB" in df.columns:
            df["TotalSizeGB"] = np.round(df["TotalSizeMB"].astype(float) / 1024, 2)
        if "FreeSpaceMB" in df.columns:
            df["FreeSpaceGB"] = np.round(df["FreeSpaceMB"].astype(float) / 1024, 2)

        # ✅ Round numeric columns for presentation
        num_cols = df.select_dtypes(include=["number"]).columns
        df[num_cols] = df[num_cols].round(2)

        return df

    except Exception as e:
        raise Exception(f"Error executing query: {e}")
    finally:
        conn.close()


# =====================================================
# 💬 MAIN CHAT WITH HISTORY
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
        with st.chat_message("user"):
            st.markdown(prompt)

        sql = map_question_to_sql(prompt)
        if not sql:
            reply = "⚠️ Sorry, I didn’t understand your question."
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        else:
            try:
                with st.spinner("Executing query..."):
                    df = run_sql(sql)
                if not df.empty:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"✅ Query executed successfully.\n\n```sql\n{sql}\n```"})
                    st.session_state.chat_history.append({"role": "assistant", "content": df})
                    with st.chat_message("assistant"):
                        st.markdown(f"✅ Query executed successfully.\n\n```sql\n{sql}\n```")
                        st.dataframe(df, use_container_width=True)
                else:
                    reply = f"ℹ️ No rows returned.\n\n```sql\n{sql}\n```"
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    with st.chat_message("assistant"):
                        st.markdown(reply)
            except Exception as e:
                reply = f"❌ Error executing query:\n{e}\n\n```sql\n{sql}\n```"
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                with st.chat_message("assistant"):
                    st.markdown(reply)
else:
    st.info("👈 Please connect to your SQL Server first.")
