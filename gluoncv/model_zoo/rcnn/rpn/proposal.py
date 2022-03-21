"""RPN proposals."""
from __future__ import absolute_import

from mxnet import autograd
from mxnet import gluon

from ....nn.bbox import BBoxCornerToCenter, BBoxClipToImage
from ....nn.coder import NormalizedBoxCenterDecoder


class RPNProposal(gluon.HybridBlock):
    """Proposal generator for RPN.

    RPNProposal takes RPN anchors, RPN prediction scores and box regression predictions.
    It will transform anchors, apply NMS (if set to true) to get clean foreground proposals.

    Parameters
    ----------
    clip : float
        Clip bounding box target to this value.
    min_size : int
        Proposals whose size is smaller than ``min_size`` will be discarded.
    stds : tuple of float
        Standard deviation to be multiplied from encoded regression targets.
        These values must be the same as stds used in RPNTargetGenerator.
    nms : boolean
        Whether to do nms.
    """

    def __init__(self, clip, min_size, stds):
        super(RPNProposal, self).__init__()
        self._box_to_center = BBoxCornerToCenter()
        self._box_decoder = NormalizedBoxCenterDecoder(stds=stds, clip=clip, convert_anchor=True)
        self._clipper = BBoxClipToImage()
        # self._compute_area = BBoxArea()
        self._min_size = min_size

    # pylint: disable=arguments-differ
    def forward(self, anchor, score, bbox_pred, img):
        """
        Generate proposals.
        """
        with autograd.pause():
            # restore bounding boxes
            roi = self._box_decoder(bbox_pred, anchor)

            # clip rois to image's boundary
            # roi = F.Custom(roi, img, op_type='bbox_clip_to_image')
            roi = self._clipper(roi, img)

            # remove bounding boxes that don't meet the min_size constraint
            # by setting them to (-1, -1, -1, -1)
            # width = roi.slice_axis(axis=-1, begin=2, end=3)
            # height = roi.slice_axis(axis=-1, begin=3, end=None)
            import mxnet as mx
            xmin, ymin, xmax, ymax = mx.np.split(roi, axis=-1, indices_or_sections=4)
            width = xmax - xmin + 1.0
            height = ymax - ymin + 1.0
            # TODO:(zhreshold), there's im_ratio to handle here, but it requires
            # add' info, and we don't expect big difference
            invalid = (width < self._min_size) + (height < self._min_size)

            # # remove out of bound anchors
            # axmin, aymin, axmax, aymax = F.split(anchor, axis=-1, num_outputs=4)
            # # it's a bit tricky to get right/bottom boundary in hybridblock
            # wrange = F.arange(0, 2560).reshape((1, 1, 1, 2560)).slice_like(
            #    img, axes=(3)).max().reshape((1, 1, 1))
            # hrange = F.arange(0, 2560).reshape((1, 1, 2560, 1)).slice_like(
            #    img, axes=(2)).max().reshape((1, 1, 1))
            # invalid = (axmin < 0) + (aymin < 0) + F.broadcast_greater(axmax, wrange) + \
            #    F.broadcast_greater(aymax, hrange)
            # avoid invalid anchors suppress anchors with 0 confidence
            score = mx.np.where(invalid, mx.np.ones_like(invalid, dtype=mx.np.float32) * -1.,score)
            invalid = mx.np.tile(invalid, reps=(1,1,4)) 
            roi = mx.np.where(invalid, mx.np.ones_like(invalid, dtype=mx.np.float32) * -1, roi)
            pre = mx.np.concatenate((score, roi), axis=-1)
            print(pre)
            return pre
