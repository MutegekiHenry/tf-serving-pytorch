import cv2
import glob
import numpy
import requests
import json
import os


headers = {"content-type": "application/json"}
img_file = "./images/image23.jpg"
inf_size = (640,640)
threshold = 0.5
save_to_dir = "/home/hmutegeki/Documents/AIR/whitefly/results/pytorch"
url = "http://localhost:8503/v1/models/whitefly:predict"

def prepare_image(img_path, img_size=inf_size):
	img = cv2.imread(img_path)
	h, w, c = img.shape
	img_resized = cv2.resize(img, img_size).astype('uint8').tolist()
	return img, img_resized, h, w


def draw_boxes(in_img, box_coords):
	for b in box_coords:
		cv2.rectangle(in_img, b[:2], b[2:4], color=(255,0,0), thickness=5)
	return in_img


# def process_predictions(pred_results, hgt, wid):
# 	# Get scores above threshold
# 	scores = [pred_results["detection_scores"].index(i) for i in pred_results["detection_scores"] if i>=0.006824]
# 	#scores = [pred_results["detection_scores"].index(i) for i in pred_results["detection_scores"] if i>=0.0068]
# 	boxes = [pred_results["detection_boxes"][i] for i in scores]
# 	true_boxes = [(int(b[1]*wid), int(b[0]*hgt), int(b[3]*wid), int(b[2]*hgt)) for b in boxes]
# 	return true_boxes

def process_predictions(pred_results, hgt, wid):
	# Get scores above threshold
	scores = [pred_results["predictions"][0]["output_1"].index(i) for i in pred_results["predictions"][0]["output_1"] if i>=0.5]
	boxes = [pred_results["predictions"][0]["output_0"][i] for i in scores]
	true_boxes = [(int(b[1]*wid), int(b[0]*hgt), int(b[3]*wid), int(b[2]*hgt)) for b in boxes]
	return true_boxes

oring_img, image_content, height, width = prepare_image(img_file)

# Create request body
body = {"instances": [image_content]}

# Post image
r = requests.post(url, data=json.dumps(body), headers = headers) 
#print(r.json())
with open("test_torch_newest2.json", "w") as json_file:
    json.dump(r.json(), json_file)

# Process results and draw bounding boxes
# output_img = draw_boxes(oring_img, process_predictions(r.json()["predictions"][0], height, width))
output_img = draw_boxes(oring_img, process_predictions(r.json(), height, width))

# Save img
cv2.imwrite(os.path.join(save_to_dir, os.path.basename(img_file).split(".")[0]+".jpg"),output_img)
