import os
try:
    from dotenv import load_dotenv
    load_dotenv(".env")  # reads MYSQL_* from Backend/.env
except Exception:
    pass

import pymysql

cfg = dict(
    host=os.getenv('MYSQL_HOST', '127.0.0.1'),
    port=int(os.getenv('MYSQL_PORT', '3306')),
    user=os.getenv('MYSQL_USER', 'autoplus'),
    password=os.getenv('MYSQL_PASSWORD', ''),
    database=os.getenv('MYSQL_DB', 'autoplus'),
    charset='utf8mb4',
)

print("Trying:", cfg)
conn = pymysql.connect(**cfg)
conn.close()
print("MySQL OK")
