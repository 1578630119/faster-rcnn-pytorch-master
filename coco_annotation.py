import json


def extract_annotation(coco_data, list_file):  # 提取xml中的信息
    d = {}
    for annotation in coco_data['annotations']:
        # 检查annotation中是否包含bbox字段（理论上应该总是包含）
        if 'bbox' in annotation:
            # bbox格式为[x, y, width, height]，其中(x, y)是左上角坐标
            bbox = annotation['bbox']
            # 将提取出的bbox转成[x1,y1,x2,y2],x1和y1是左上角坐标，x2和y2是右下角坐标
            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            # 将图像ID和类别ID与边界框一起存储，以便于后续处理
            category_id = annotation['category_id']
            bbox.append(category_id)
            image_id = annotation['image_id']
            if image_id not in d.keys():
                d[image_id] = []
            d[image_id].append(bbox)  # 汇聚属于同一图片的标注信息
    lines = list(d.keys())
    for line in lines:
        bboxes = d[line]
        filename = '{:012}.jpg'.format(line)
        for b in bboxes:
            filename = filename + " " + ",".join([str(int(a)) for a in b])
        list_file.write(filename + '\n')  # 写入信息


if __name__ == "__main__":
    # coco数据集版本，以COCO2014为例
    coco_year = 2014

    # 加载JSON文件
    with open('annotations/instances_train{}.json'.format(coco_year), 'r') as f:
        coco_data = json.load(f)
    with open('train.txt', 'w') as g:
        extract_annotation(coco_data, g)

    with open('annotations/instances_val{}.json'.format(coco_year), 'r') as f:
        coco_data = json.load(f)
    with open('val.txt', 'w') as g:
        extract_annotation(coco_data, g)

    # 加载JSON文件
    with open('annotations/instances_test{}.json'.format(coco_year), 'r') as f:
        coco_data = json.load(f)
    with open('test.txt', 'w') as g:
        extract_annotation(coco_data, g)