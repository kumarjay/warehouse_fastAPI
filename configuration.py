from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def configuration_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # cfg.DATASETS.TRAIN = ("/var/warehouse/resized/images/train",)
    # cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 2
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 5000
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.MODEL.DEVICE = 'cpu'

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    # cfg.DATASETS.TEST = ('/var/warehouse/resized/images/test',)

    # cfg.MODEL.WEIGHTS = os.path.join("/var/home_", "where model is saved")
    cfg.MODEL.WEIGHTS = 'model_final.pth'
    predictor = DefaultPredictor(cfg)
    return predictor
