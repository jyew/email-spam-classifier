FROM python:3.7.7-slim

ENV PYTHONUNBUFFERED=1

COPY * /opt/microservices/
COPY requirements.txt /opt/microservices/

WORKDIR /opt/microservices/

RUN pip install --upgrade pip \
  && pip install --upgrade pipenv\
  && apt-get clean \
  && apt-get update \
  && apt-get -y install wget unzip && rm -rf /var/cache/apk/* \
  && wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1i2I7ulyoa9bJRFGOFhglR7c2E27peFz2' -O models.zip \
  && unzip models.zip -d models \
  && rm models.zip \
  && apt install -y build-essential \
  && apt install -y libmariadb3 libmariadb-dev \
  && pip install --upgrade -r /opt/microservices/requirements.txt \
  && mkdir -p /nltk_data \
  && chown 1001:1001 -R /nltk_data \ 
  && chmod -R 777 /nltk_data

USER 1001

EXPOSE 8080

CMD ["python", "app.py", "8080"]