FROM ubuntu:20.04
RUN apt update -y
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y 
RUN apt install python3.8 -y
RUN apt update -y 
RUN apt install python3-pip -y
RUN mkdir -p /usr/app
WORKDIR /usr/app
COPY . /usr/app
RUN pip install -U opencv-python
RUN apt -y  upgrade
RUN apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx
RUN pip install -r requirements.txt
CMD ["python3","app.py"]
