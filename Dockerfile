FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install fastapi uvicorn torch transformers numpy tqdm tensorboardX

EXPOSE 80

COPY ./app /app
COPY ./data /data
COPY ./in /in
COPY ./out /out
COPY ./roberta-base /roberta-base

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "80"]
