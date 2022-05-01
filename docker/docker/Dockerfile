FROM armindocachada/tensorflow2-opencv4-raspberrypi4:2.2_4.5.0 AS base

FROM base as build

RUN apt-get --allow-releaseinfo-change update

RUN apt-get -y upgrade

RUN python3 -m pip install --upgrade pip

RUN apt-get -y install python3-pandas

RUN apt-get -y install python3-sklearn python3-sklearn-lib

RUN apt-get -y install mc


