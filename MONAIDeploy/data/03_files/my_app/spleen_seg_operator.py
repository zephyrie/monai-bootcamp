import logging
from os import path

from numpy import uint8

import monai.deploy.core as md
from monai.deploy.core import ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
from monai.deploy.operators.monai_seg_inference_operator import InMemImageReader, MonaiSegInferenceOperator
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)


@md.input("image", Image, IOType.IN_MEMORY)
@md.output("seg_image", Image, IOType.IN_MEMORY)
@md.env(pip_packages=["monai==0.6.0", "torch>=1.5", "numpy>=1.20", "nibabel", "typeguard"])
class SpleenSegOperator(Operator):
    """Performs Spleen segmentation with a 3D image converted from a DICOM CT series.
    """

    def __init__(self):

        self.logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__()
        self._input_dataset_key = "image"
        self._pred_dataset_key = "pred"

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):

        input_image = op_input.get("image")
        if not input_image:
            raise ValueError("Input image is not found.")

        output_path = context.output.get().path

        # This operator gets an in-memory Image object, so a specialized ImageReader is needed.
        _reader = InMemImageReader(input_image)
        pre_transforms = self.pre_process(_reader)
        post_transforms = self.post_process(pre_transforms, path.join(output_path, "prediction_output"))

        # Delegates inference and saving output to the built-in operator.
        infer_operator = MonaiSegInferenceOperator(
            (
                160,
                160,
                160,
            ),
            pre_transforms,
            post_transforms,
        )

        # Setting the keys used in the dictironary based transforms may change.
        infer_operator.input_dataset_key = self._input_dataset_key
        infer_operator.pred_dataset_key = self._pred_dataset_key

        # Now let the built-in operator handles the work with the I/O spec and execution context.
        infer_operator.compute(op_input, op_output, context)

    def pre_process(self, img_reader) -> Compose:
        """Composes transforms for preprocessing input before predicting on a model."""

        my_key = self._input_dataset_key
        return Compose(
            [
                LoadImaged(keys=my_key, reader=img_reader),
                EnsureChannelFirstd(keys=my_key),
                Spacingd(keys=my_key, pixdim=[1.0, 1.0, 1.0], mode=["blinear"], align_corners=True),
                ScaleIntensityRanged(keys=my_key, a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=my_key, source_key=my_key),
                ToTensord(keys=my_key),
            ]
        )

    def post_process(self, pre_transforms: Compose, out_dir: str = "./prediction_output") -> Compose:
        """Composes transforms for postprocessing the prediction results."""

        pred_key = self._pred_dataset_key
        return Compose(
            [
                Activationsd(keys=pred_key, softmax=True),
                AsDiscreted(keys=pred_key, argmax=True),
                Invertd(
                    keys=pred_key, transform=pre_transforms, orig_keys=self._input_dataset_key, nearest_interp=True
                ),
                SaveImaged(keys=pred_key, output_dir=out_dir, output_postfix="seg", output_dtype=uint8, resample=False),
            ]
        )
