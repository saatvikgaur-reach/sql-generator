import pandas as pd
import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import requests
import sys
from termcolor import colored
import duckdb
import colorama
import re
import datetime

colorama.init()

def clean_data(file_path, nrows=None, normalize_headers=True):
    """
    Reads an events file, removes duplicate rows and empty columns, normalizes column headers,
    splits datetime columns into separate date and time columns, and returns the cleaned DataFrame.
    Args:
        file_path (str): Path to the events CSV file.
        nrows (int, optional): Number of rows to read from the CSV. If None, reads all rows.
        normalize_headers (bool): Whether to normalize column headers (default: True).
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """

    df = pd.read_csv(file_path, low_memory=False, nrows=nrows)
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how='all')

    if normalize_headers:
        def normalize_column(col):
            col = col.strip().lower()
            col = re.sub(r'[^\w]+', '_', col)
            col = re.sub(r'__+', '_', col)
            return col.strip('_')
        # Normalize and ensure uniqueness
        new_cols = [normalize_column(col) for col in df.columns]
        # Make unique if needed
        seen = {}
        unique_cols = []
        for col in new_cols:
            if col not in seen:
                seen[col] = 1
                unique_cols.append(col)
            else:
                seen[col] += 1
                unique_cols.append(f"{col}_{seen[col]}")
        df.columns = unique_cols

    # Detect and split datetime-like columns (keep as strings)
    datetime_pattern = re.compile(r"\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+[AP]M")

    for col in df.columns:
        col_data = df[col]
        # If col_data is a DataFrame (duplicate columns), skip or warn
        if isinstance(col_data, pd.DataFrame):
            print(f"Warning: Duplicate column detected and skipped: {col}")
            continue
        if col_data.dtype == 'object':
            sample = col_data.dropna().astype(str)
            if not sample.empty:
                sample_val = sample.iloc[0]
                if datetime_pattern.fullmatch(sample_val.strip()):
                    date_part = col_data.str.extract(r"^(\d{2}/\d{2}/\d{4})")[0]
                    time_part = col_data.str.extract(r"\s+(\d{1,2}:\d{2}:\d{2}\s+[AP]M)$")[0]
                    df[f"{col}_date"] = date_part
                    df[f"{col}_time"] = time_part
                    df.drop(columns=[col], inplace=True)

    # Convert all string data in the DataFrame to lowercase
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower()

    return df


get_path = lambda file="": os.path.join((os.path.dirname(os.path.abspath(__file__))), file)
try:
    print("Retrieving Events ...")
    events = pd.read_csv(get_path("events_clean.csv"), low_memory=False)
except:
    print("No events_clean.csv found, cleaning data ...")
    print("Cleaning Data ...")
    events = clean_data(get_path("events.csv"))
    events.to_csv('events_clean.csv', index=False) 

column_headers = events.columns.tolist()

dictionary_path = get_path("dictionary.txt")
with open(dictionary_path, "r", encoding="utf-8") as f: data_dictionary = f.read()


def generate(prompt, model, stream=False):
    """
    Generates a response from the sqlcoder model via Ollama's local API.
    Args:
        prompt (str): The prompt to send to the model.
        model (str): The model name in Ollama.
        stream (bool): Whether to use streaming responses.
    Returns:
        str: The generated response (SQL code as plain text).
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {"temperature": 0}
    }
    response = requests.post(url, json=payload)
    if stream:
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))
        return None
    else:
        # sqlcoder returns the SQL code as plain text in the 'response' key
        result = response.json()
        return result.get("response", result)


def extract_last_sql_code(text):
    """
    Extracts the last SQL code chunk from a string. Handles markdown code blocks, multiple code blocks, and fallback to last SELECT/WITH.
    Args:
        text (str): The text to extract SQL from.
    Returns:
        str: The extracted SQL code, or None if not found.
    """
    # 1. Find all markdown SQL code blocks
    code_blocks = re.findall(r"```sql\s*([\s\S]+?)```", text, re.IGNORECASE)
    if code_blocks:
        # Return the last code block, stripped
        return code_blocks[-1].strip()
    
    # 2. Fallback: Find all SELECT/WITH statements ending with semicolon
    matches = list(re.finditer(r"((SELECT|WITH)[\s\S]+?;)\s*", text, re.IGNORECASE))
    if matches:
        # Return the last match
        return matches[-1].group(1).strip()
    
    # 3. Fallback: Find the last occurrence of SELECT or WITH and return to end
    last_select = max(text.lower().rfind('select'), text.lower().rfind('with'))
    if last_select != -1:
        # Try to get to the end of the statement (up to next blank line or end)
        after = text[last_select:]
        # Stop at first double newline or end
        after = after.split('\n\n')[0]
        return after.strip()
    
    # 4. As a last resort, return the whole text
    return text.strip()


def answer_question(question, max_rows=10):
    """
    Answers a question about the trucking events using the sqlcoder model via Ollama.
    Args:
        question (str): The user's question.
        chunks (List[dict]): The event data as a list of dictionaries.
        max_rows (int): Maximum number of rows to include in the prompt for data.
    Returns:
        str: The model's answer (SQL code).
    """
    prompt_path = get_path("prompt.txt")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    prompt = prompt_template.format(column_headers=column_headers, data_dictionary=data_dictionary, question=question)

    response = generate(prompt, model="deepseek-r1:7b")
    answer = response if isinstance(response, str) else str(response)

    # Use the new extraction function
    sql_code = extract_last_sql_code(answer)

    # Post-process the SQL code
    sql_code = re.sub(
        r"\((\w+_date) \|\| ' ' \|\| (\w+_time)\)::timestamp",
        r"STRPTIME(\1 || ' ' || \2, '%m/%d/%Y %I:%M:%S %p')",
        sql_code
    )

    return sql_code, answer

def queryEvents(query):
    con = duckdb.connect()
    try:
        # Use the in-memory DataFrame 'events' as the 'events' table in DuckDB
        con.register('events_clean', events)
        result = con.execute(query).fetchdf()
        return "SQL Query Result: ", result
    except Exception as e:
        return ("Error running SQL:", e)

while True:
    question = input("Question: ")
    if question == "exit" or question == "Exit": 
        break
    response, reasoning = answer_question(question)
    print()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Question:", question)
    print("Reasoning: ", reasoning)
    print(colored("Running:", 'magenta'))
    print(colored(response, 'magenta'))
    print()
    print(colored(queryEvents(response), 'cyan'))
