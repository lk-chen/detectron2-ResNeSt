from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
import cv2
import os

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = os.path.join("output/model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    # im = cv2.imread("datasets/VOC2012/JPEGImages/000052.png")  # many cars
    # im = cv2.imread("datasets/VOC2012/JPEGImages/000043.png")  # many ped
    # im = cv2.imread("datasets/VOC2012/JPEGImages/007318.png")  # many cycle
    im = cv2.imread("datasets/VOC2012/JPEGImages/004384.png")  # many cycle
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    print(f'class names are: {metadata.get("thing_classes")}')
    pred_class = outputs["instances"].pred_classes
    print(f'pred_classes are: {pred_class}')
    readable_pred_class = [metadata.get("thing_classes")[idx] for idx in pred_class.tolist()]
    print(f'Human readable pred_class: {readable_pred_class}')
    print(outputs["instances"].pred_boxes)
    print(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    v = Visualizer(im[:, :, ::-1], metadata, scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    v.save("prediction.png")



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
