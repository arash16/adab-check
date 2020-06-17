FROM tensorflow/tensorflow:2.2.0
RUN pip install setuptools-rust gdown
RUN pip --version
WORKDIR /app

RUN gdown https://drive.google.com/uc?id=1-BZ4HWBXwz9HS4HUopnq0yLrw26QZFGq -O model_weights.h5

ADD requirements.txt /app/
RUN pip install -r requirements.txt

ADD fetch-models.py /app/
RUN python /app/fetch-models.py

ADD . /app

HEALTHCHECK --interval=20s --timeout=10s --retries=3 CMD curl -sS 127.0.0.1 || exit 1
CMD ["sh", "run.sh"]
