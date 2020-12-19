from fastapi import FastAPI, UploadFile, File, requests, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
#from starlette.responses import FileResponse
from fastapi.responses import FileResponse
import os
import shutil
import pymongo

from warehouse_box import Box
from configuration import configuration_model
# import warehouse_box as box

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import itertools
import pandas as pd
import os, cv2

from starlette.responses import Response

# from flask import render_template, request, redirect

from typing import List
from io import BytesIO
from PIL import Image
# from config import PORT
import uvicorn
from pathlib import Path

from pydantic import BaseModel

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLALCHEMY_DATABASE_URL = "mongodb+srv://user:user@cluster0.2jv3l.mongodb.net/warehouse_01?retryWrites=true&w=majority"
# # SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"
#
# engine = create_engine(
#     SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
# )
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#
# Base = declarative_base()

app = FastAPI()

dbAtlas= pymongo.MongoClient("mongodb+srv://user:user@cluster0.2jv3l.mongodb.net/warehouse_01?retryWrites=true&w=majority")
db= dbAtlas.get_database('warehouse_01')
records= db.object_detection_01

classes= ['Pallet Jacks', 'Rolling Ladder', 'Wire Mesh', 'Bulk Box', 'Totes',
       'Dump Hopper', 'Bin', 'Yard Ramp']

# for d in ["train", "test"]:
#     DatasetCatalog.register("experiment1/" + d, lambda d=d: Box.get_warehouse_box("/var/warehouse/resized/images/data_14.csv", "//var/warehouse/resized/images/"+ d +"/"))
#     MetadataCatalog.get("experiment1/" + d).set(thing_classes=classes)
warehouse_metadata = MetadataCatalog.get("experiment1/train").set(thing_classes=classes)
print('metadata...', warehouse_metadata)
print('metadata....', os.getcwd())


db = []
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')


# app.mount(
#     "/static",
#     StaticFiles(directory=Path(__file__).parent.parent.absolute() / "static"),
#     name="static",
# )


class City(BaseModel):
    name: str
    timezone: str  # list, datetime, dict


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request, 'img': 2})


@app.get('/xyz', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('abc.html', {'request': request})


@app.get('/cities')
def get_cities():
    return db


@app.get('/cities/{city_id}')
def get_city(city_id: int):
    return db[city_id - 1]


#
@app.post('/cities')
def create_city(city: City):
    db.append(city.dict())
    return db[-1]


#
@app.delete('/cities/{city_id}')
def delete_city(city_id: int):
    db.pop(city_id - 1)
    return {}


@app.post('/upload-image')
def create_upload_files(image: UploadFile = File(...)):
    # form = await request.form()
    print('Hello World')
    print('something.....', image.filename)
    print('abccc....', image)
    # with open(form.values(), 'r') as f:
    #     xyz = f.write()
    temp_file= _save_file_to_disk(image,  path='static/assets/img/searched_image', save_as=image.filename)

    img = cv2.imread('static/assets/img/searched_image/'+image.filename)
    # print('image name....', self.window.filename)
    print('image shape is.....', img.shape)

    predictor = configuration_model()

    output = predictor(img)
    visualizer = Visualizer(img[:, :, ::-1], metadata=warehouse_metadata, scale=0.5)
    out = visualizer.draw_instance_predictions(output['instances'].to('cpu'))
    img_out = Image.fromarray(out.get_image()[:, :, ::-1])

    prediction= output['instances'].pred_classes.numpy()
    dict_list= list(set(prediction))
    dict_= {}

    for name_ in dict_list:
        dict_[classes[name_]] = 0
    print('dict_ value.....', dict_)

    for name_ in prediction:
        dict_[classes[name_]] = dict_[classes[name_]]+1
    image_1= {'Image': image.filename}

    dict_['Image']= image.filename
    records.insert_one(dict_)
    print('predicted.......', out, 'and.....', img_out)
    print('instances.....', output['instances'])
    print('dictionary is....', dict_)

    print('database is......', records.count_documents({}), type(dict_))
    #abc = dict_
    #print(abc, type(abc))


    # cv2.imshow('abc', out.get_image()[:, :, ::-1])
    # cv2.imshow('xyz', img)
    cv2.imwrite(f'static/assets/img/predicted_image/{image.filename}', out.get_image()[:, :, ::-1])

    # temp_file = _save_file_to_disk(out.get_image()[:, :, ::-1], path='static/assets/img/predicted_image', save_as=image.filename+'_pred')
    xyzz = FileResponse("static/assets/img/predicted_image/pred.jpg")
    # pil_img= Image.open(BytesIO(form.values()[0]))
    # print('filename is.....', xyz)
    dict_.pop('_id')
    return {'text': image.filename, 'key': list(dict_.keys()), 'value': list(dict_.values()), 'image': xyzz }


def _save_file_to_disk(uploaded_file, path=".", save_as="default"):
    extension = os.path.splitext(uploaded_file.filename)[-1]
    temp_file = os.path.join(path, save_as)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return temp_file


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
