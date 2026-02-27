@echo off
title data_checker

echo [INFO] 正在檢查環境...

echo [INFO] 啟動 Streamlit...
python -m streamlit run app.py --server.port 8501

echo.
echo [結束] 應用程式已關閉，按任意鍵離開...
pause