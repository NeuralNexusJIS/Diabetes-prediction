FROM python:3.11
COPY . /app


WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 7000    
CMD streamlit run app.py --server.port 7000 --server.enableCORS false
