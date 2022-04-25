FROM python:3
ADD . /
EXPOSE 5000
WORKDIR /

RUN pip3 install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

ENTRYPOINT ["python"] 
CMD ["app-test.py"]
