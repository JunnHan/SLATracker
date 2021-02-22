from detectron2.config import CfgNode
from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C

def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    # _C.MODEL.RPN.NMS_THRESH = 0.7

    _C.MODEL.ROI_BOX_HEAD.POOLER_TYPE = 'PrROIPool'
    _C.SOLVER.CHECKPOINT_PERIOD = 100000
    _C.MODEL.ROI_HEADS.NUM_CLASSES = 1

    _C.MODEL.REID_HEADS = CN()
    _C.MODEL.REID_HEADS.NAME = 'REID_HEAD'
    _C.MODEL.REID_HEADS.FG_FRACTION = 0.5  # Fraction of minibatch that is labeled foreground (i.e. class > 0)
    _C.MODEL.REID_HEADS.FG_THRESH = 0.5 # Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)

    _C.MODEL.IOU_HEADS = CN()
    _C.MODEL.IOU_HEADS.NAME = 'IOU_HEAD'
    _C.MODEL.IOU_HEADS.NMS_THRESH = 0.5

    # Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))
    _C.MODEL.REID_HEADS.BG_THRESH_HI = 0.5
    _C.MODEL.REID_HEADS.BG_THRESH_LO = 0.1

    return _C.clone()
