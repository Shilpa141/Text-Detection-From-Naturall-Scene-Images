from shapely.geometry import Polygon
import os
import cv2
import numpy as np

actual_path = "Dataset/Actual/"
predicted_path = "Dataset/Results/"
iou_filename = "Dataset/iou.txt"

def IOU(pol1_xy, pol2_xy):
    # Define each polygon
    polygon1_shape = Polygon(pol1_xy)
    polygon2_shape = Polygon(pol2_xy)

    # Calculate intersection and union, and the IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.union(polygon2_shape).area
    return polygon_intersection / polygon_union


false_negative = 0
true_positive = 0
false_positive = 0
file1 = open(iou_filename, "w")
for file in os.listdir(actual_path):
    if file.endswith('.txt'):
        file_exists = os.path.exists(predicted_path + file)
        if file_exists:
            print(file)
            f1 = open(actual_path+file, "r")
            f2 = open(predicted_path+file, "r")
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            line_count = 0
            flag = 0
            true_positive_in_file = 0
            for element1 in lines1:
                coordinates1 = element1.split(",")
                coordinates1 = coordinates1[:8]
                coordinates1 = list(map(int, coordinates1))
                # print(coordinates1)
                pol1_xy = [[coordinates1[0], coordinates1[1]], [coordinates1[2], coordinates1[3]],
                           [coordinates1[4], coordinates1[5]], [coordinates1[6], coordinates1[7]]]
                max_iou = 0
                true_coordinates = []

                for element2 in lines2:
                    if flag == 0:
                        line_count += 1
                    coordinates2 = list(map(int, element2.split(",")))
                    # print(coordinates2)
                    pol2_xy = [[coordinates2[0], coordinates2[1]], [coordinates2[2], coordinates2[3]],
                               [coordinates2[4], coordinates2[5]], [coordinates2[6], coordinates2[7]]]
                    iou_value = IOU(pol1_xy, pol2_xy)
                    if max_iou < iou_value and iou_value >= 0.5 :
                        max_iou = iou_value
                        true_coordinates = pol2_xy

                if max_iou >= 0.5:
                    line = [file.replace(".txt", ""), pol1_xy, true_coordinates, "IOU: "+str(max_iou)]
                    true_positive += 1
                    true_positive_in_file += 1
                else:
                    line = [file.replace(".txt", ""), pol1_xy, "IOU: -1"]
                    false_negative += 1
                flag = 1
                image_file = predicted_path+file.replace(".txt",".jpg")
                if os.path.exists(image_file):
                    img = cv2.imread(image_file)
                    pts = np.array(pol1_xy)
                    pts.reshape((-1,1,2))
                    image = cv2.polylines(img, [pts], True, (255, 0, 0), 2)
                    cv2.putText(image, "IoU: {:.2f}".format(max_iou),
                                ((pol1_xy[0][0], pol1_xy[0][1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                    cv2.imwrite(image_file, image)

                print(line)
                file1.write(str(line))
                file1.write("\n")

            false_positive += line_count - true_positive_in_file
        else:
            print(file+" doesn't exist")
            f1 = open(actual_path + file, "r")
            lines1 = f1.readlines()
            for element1 in lines1:
                coordinates1 = element1.split(",")
                coordinates1 = coordinates1[:8]
                coordinates1 = list(map(int, coordinates1))
                # print(coordinates1)
                pol1_xy = [[coordinates1[0], coordinates1[1]], [coordinates1[2], coordinates1[3]],
                           [coordinates1[4], coordinates1[5]], [coordinates1[6], coordinates1[7]]]
                line = [file.replace(".txt", ""), pol1_xy, "IOU: -1"]
                print(line)
                file1.write(str(line))
                file1.write("\n")
                false_negative += 1

print("True Positive: ",  true_positive)
print("False Negative: ", false_negative)
print("False Positive: ", false_positive)
precision=true_positive/(true_positive+false_positive)
recall=true_positive/(true_positive+false_negative)
f_score=2*(precision*recall)/(precision+recall)
print("Accuracy: ",true_positive/(true_positive+false_negative+false_positive) )
print(f"Precision:{precision} \t Recall:{recall} \t F_score:{f_score}") 


