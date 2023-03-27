import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np


ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )
    tmp = 0
    val_tot = 0
    while True:
        ret, frame = cap.read()
        try:
            height = frame.shape[0]
            width = frame.shape[1]            
        except:
            pass
        cross = 0
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)        
        detections = detections[detections.class_id == 76]        
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        print(len(labels))        
        if(len(labels)):
            tmp = 1
        else:
            tmp = 0
        if(tmp == 1):
            val_tot = val_tot+1
        cv2.putText(frame,"Scissors "+str(val_tot), (200,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
        if("cell phone" in labels):
            pass
        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)      
        
        cv2.imshow("yolov8", frame)
        out.write(frame) 
        if (cv2.waitKey(30) == 27):
            print("total Scissors"+str(val_tot))
            break
    cap.release()
    out.release()


if __name__ == "__main__":
    main()