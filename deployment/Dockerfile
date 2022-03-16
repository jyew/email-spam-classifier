FROM python:3.7.7-slim

ENV PYTHONUNBUFFERED=1

COPY * /opt/microservices/
COPY requirements.txt /opt/microservices/
RUN pip install --upgrade pip \
  && pip install --upgrade pipenv\
  && apt-get clean \
  && apt-get update \
  && apt install -y build-essential \
  && apt install -y libmariadb3 libmariadb-dev \
  && apt-get -y install wget unzip && rm -rf /var/cache/apk/* \
  && wget https://rainbowtechcomhk-my.sharepoint.com/:u:/g/personal/jordan_yew_rainbowtech_com_hk/EdJU_x2NtatKs5mue3qHOIcBE4vmuDEj1UDCeVZCMVUSDA?e=oVgG1H -P models/ \
  && wget https://rainbowtechcomhk-my.sharepoint.com/:u:/g/personal/jordan_yew_rainbowtech_com_hk/EZKQLmuMMbNJiEYamMcSRoIBvS-IfXfs0lSviyap11V_5Q?e=V6rgb6 -P models/ \
  && pip install --upgrade -r /opt/microservices/requirements.txt

USER 1001

EXPOSE 8080
WORKDIR /opt/microservices/

CMD ["python", "app.py", "8080"]