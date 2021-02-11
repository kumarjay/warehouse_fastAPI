FROM ubuntu:18.04

#ENV PATH="/root/.local/bin:${PATH}"
#ARG PATH="/root/.local/bin:${PATH}"

ENV export LC_ALL=C.UTF-8
ENV export LANG=C.UTF-8

EXPOSE 8000

RUN apt-get update
RUN apt-get install --assume-yes --fix-broken
RUN apt-get install -y python3-dev wget git
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py
RUN apt-get install python3-pip -y
RUN apt-get install -y gcc

RUN pip3 install --upgrade pip
RUN pip3 install scikit-build
COPY . src/
WORKDIR /src/
RUN  pip3 install -r requirements.txt
RUN pip3 install  'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN pip3 install -e detectron2_repo
RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN locale-gen en_US.UTF-8
# ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
