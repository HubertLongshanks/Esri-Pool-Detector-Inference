import torch
from torch import Tensor
import torchvision as tv
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import rasterio as rs
from rasterio.transform import Affine
import torch
import os
import math
from torchvision.ops import batched_nms
from torchgeo.datasets import RasterDataset
from torchgeo.samplers import GridGeoSampler
from torch.utils.data import DataLoader
from dataclasses import dataclass
import geopandas as gpd
import shapely


@dataclass
class PoolDetectorResults:
    """A class encapsulating the return values of the results of a pool detector run over a raster image, returned bounding boxes will be in the CRS of the inference file and correspond to the most confident bbox of the model for that detection."""

    inference_file: str
    bboxes: list[Tensor]
    confidences: list[Tensor]


def collate(batch):
    images = torch.stack([ex["image"] for ex in batch], dim=0)
    return {"images": images, "bounds": [ex["bounds"] for ex in batch]}


def boxes_to_global(merged: dict, chip_width: int, chip_height: int) -> Tensor:
    """
    Converts a batch of local Faster R-CNN bounding boxes to global coordinates.
    """
    all_global_boxes = []

    # Iterate through each image's results and bounds in the batch
    for model_res, bounds in zip(merged["model_results"], merged["bounds"]):
        boxes = model_res["boxes"]

        if len(boxes) == 0:
            continue

        # Extract global min/max for this specific chip
        x_min_g, x_max_g = bounds[0].start, bounds[0].stop
        y_min_g, y_max_g = bounds[1].start, bounds[1].stop

        # Calculate resolution
        x_res = (x_max_g - x_min_g) / chip_width
        y_res = (y_max_g - y_min_g) / chip_height

        # Transform coordinates
        global_boxes = torch.empty_like(boxes)
        global_boxes[:, 0] = x_min_g + (boxes[:, 0] * x_res)
        global_boxes[:, 2] = x_min_g + (boxes[:, 2] * x_res)
        global_boxes[:, 1] = y_max_g - (boxes[:, 3] * y_res)
        global_boxes[:, 3] = y_max_g - (boxes[:, 1] * y_res)

        all_global_boxes.append(global_boxes)

    # If no boxes were found in the entire batch, return an empty tensor
    if not all_global_boxes:
        return torch.empty((0, 4))

    # Return flattened boxes for the batch
    return torch.concat(all_global_boxes, dim=0)


class EsriPoolFasterRCNNInference:
    def __init__(
        self,
        raster_mosaic_path: str,
        model_weights_path: str,
        detection_confidence_score: float = 0.6,
        expected_model_img_size: int = 224,
        window_stride_fraction: float = 1 / 2,
        batch_size=16,
        nodata: int = 0,
    ):

        assert os.path.exists(
            raster_mosaic_path
        ), f"{raster_mosaic_path} does not exist"
        assert (
            detection_confidence_score >= 0 and detection_confidence_score <= 1
        ), "cnfidence outside of range [0,1]"

        self.model = fasterrcnn_resnet50_fpn(
            num_classes=2, box_score_thresh=detection_confidence_score
        )

        state = torch.load(f=model_weights_path, map_location=torch.device("cpu"))

        self.model.load_state_dict(state_dict=state, strict=True)

        # self.model.compile()

        self.detection_threshold = detection_confidence_score
        self.raster_mosaic = raster_mosaic_path
        self.img_resolution = expected_model_img_size
        self.batch_size = batch_size
        self.stride = window_stride_fraction
        self.nodata = nodata

    def perform_inference(
        self,
        device: str = "cuda",
        num_dataworkers: int = max(os.cpu_count() - 2, 1),
        detections_file: str | None = None,
    ) -> PoolDetectorResults:
        """Performs inference on the raster mosaic specified and returns results as bboxes in global CRS units

        Args:
            device (str, optional): device to run infenence on. Defaults to "cuda".
            num_dataworkers (int, optional): number of workers to use to load data. Defaults to max(os.cpu_count() - 2, 1).
            detections_file : str | None (optional): the file to write detections for the raster to, can be any format supported by geopandas, e.g. .shp, .geojson, .fgb
        Returns:
            PoolDetectorResults: the results of the inference after NMS and casting to global coordinates
        """

        with rs.open(self.raster_mosaic) as raster_dataset:

            file_transform: Affine = raster_dataset.transform

            file_size: tuple[int, int] = (raster_dataset.width, raster_dataset.height)

            # current_offset : tuple[int,int] = ( 0 , 0 ) #column, row fo can *cuur_offset and works

            inference_results: list = []

            self.model = self.model.to(device=device)
            self.model.eval()

            dataset = RasterDataset(paths=self.raster_mosaic)

            sampler = GridGeoSampler(
                dataset=dataset,
                size=(self.img_resolution, self.img_resolution),
                stride=(
                    math.floor(self.img_resolution * self.stride),
                    math.floor(self.img_resolution * self.stride),
                ),
            )

            loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=collate,
                num_workers=num_dataworkers,
                pin_memory=True if torch.cuda.is_available() else False,
                persistent_workers=False,
            )

            for batch in loader:
                examples = batch["images"].to(device=device) / 255.0

                with torch.inference_mode():
                    results: list[dict[str, Tensor]] = self.model(examples)

                    # clear gpu memory and preform nms? or global later?
                    if device != "cpu":
                        for res in results:
                            for idx, key in enumerate(res):
                                res[key] = res[key].detach().cpu()

                    inference_results.append(
                        {"model_results": results, "bounds": batch["bounds"]}
                    )

            merged = {
                "boxes": torch.empty((0, 4)),
                "labels": torch.tensor([]),
                "scores": torch.tensor([]),
            }

            # now combine and preform NMS for the entire mosaic

            for res in inference_results:

                global_shifted_bboxes = boxes_to_global(
                    merged=res,
                    chip_width=self.img_resolution,
                    chip_height=self.img_resolution,
                )

                merged["boxes"] = torch.concat([merged["boxes"], global_shifted_bboxes])
                merged["labels"] = torch.concat(
                    [
                        merged["labels"],
                        torch.concat(
                            [i["labels"] for i in res["model_results"]]
                        ).flatten(),
                    ]
                )
                merged["scores"] = torch.concat(
                    [
                        merged["scores"],
                        torch.concat(
                            [i["scores"] for i in res["model_results"]]
                        ).flatten(),
                    ]
                )

            # preform global NMS
            keep = batched_nms(
                boxes=merged["boxes"],
                scores=merged["scores"],
                idxs=merged["labels"],
                iou_threshold=0.20,
            )

            merged["boxes"] = merged["boxes"][keep]
            merged["scores"] = merged["scores"][keep]
            merged["labels"] = merged["labels"][keep]

            if detections_file != None:
                bbox_polys = [
                    shapely.geometry.box(*(i.numpy())) for i in merged["boxes"]
                ]
                scores = merged["scores"].numpy()

                gdf = gpd.GeoDataFrame(
                    data=scores, columns=["scores"], geometry=bbox_polys
                )

                gdf.set_crs(raster_dataset.crs, inplace=True)

                gdf.to_file(detections_file)

            return PoolDetectorResults(
                inference_file=self.raster_mosaic,
                bboxes=merged["boxes"],
                confidences=merged["scores"],
            )
