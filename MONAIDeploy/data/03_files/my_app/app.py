import logging

from spleen_seg_operator import SpleenSegOperator

from monai.deploy.core import Application, resource
from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
from monai.deploy.operators.dicom_seg_writer_operator import DICOMSegmentationWriterOperator
from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator

@resource(cpu=1, gpu=1, memory="7Gi")
class AISpleenSegApp(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compose(self):

        study_loader_op = DICOMDataLoaderOperator()
        series_selector_op = DICOMSeriesSelectorOperator(Sample_Rules_Text)
        series_to_vol_op = DICOMSeriesToVolumeOperator()
        # Creates DICOM Seg writer with segment label name in a string list
        dicom_seg_writer = DICOMSegmentationWriterOperator(seg_labels=["Spleen"])
        # Creates the model specific segmentation operator
        spleen_seg_op = SpleenSegOperator()

        # Creates the DAG by link the operators
        self.add_flow(study_loader_op, series_selector_op, {"dicom_study_list": "dicom_study_list"})
        self.add_flow(series_selector_op, series_to_vol_op, {"study_selected_series_list": "study_selected_series_list"})
        self.add_flow(series_to_vol_op, spleen_seg_op, {"image": "image"})
        self.add_flow(series_selector_op, dicom_seg_writer, {"study_selected_series_list": "study_selected_series_list"})
        self.add_flow(spleen_seg_op, dicom_seg_writer, {"seg_image": "seg_image"})

# This is a sample series selection rule in JSON, simply selecting CT series.
# If the study has more than 1 CT series, then all of them will be selected.
# Please see more detail in DICOMSeriesSelectorOperator.
Sample_Rules_Text = """
{
    "selections": [
        {
            "name": "CT Series",
            "conditions": {
                "StudyDescription": "(.*?)",
                "Modality": "(?i)CT",
                "SeriesDescription": "(.*?)"
            }
        }
    ]
}
"""

if __name__ == "__main__":
    # Creates the app and test it standalone. When running is this mode, please note the following:
    #     -i <DICOM folder>, for input DICOM CT series folder
    #     -o <output folder>, for the output folder, default $PWD/output
    #     -m <model file>, for model file path
    # e.g.
    #     python3 app.py -i input -m model.ts
    #
    AISpleenSegApp(do_run=True)
