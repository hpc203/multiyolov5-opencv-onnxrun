import cv2
import argparse
import numpy as np
import onnxruntime as ort

Cityscapes_Class = ["road", "sidewalk", "building", "wall", "fence",
               "pole", "traffic light", "traffic sign", "vegetation",
               "terrain", "sky", "person", "rider", "car", "truck",
               "bus", "train", "motorcycle", "bicyle"]
Cityscapes_COLORMAP = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [0, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]

class yolov5():
    def __init__(self, modelpath, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        with open('class.names', 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        num_classes = len(self.classes)
        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.inpWidth = 1024
        self.inpHeight = 1024
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(modelpath, so)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold

    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.inpWidth, self.inpHeight
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.inpWidth - neww - left, cv2.BORDER_CONSTANT,
                                         value=(114, 114, 114))  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.inpHeight - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left
    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)
    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def postprocess(self, frame, outs, padsize=None):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        newh, neww, padh, padw = padsize
        ratioh, ratiow = frameHeight / newh, frameWidth / neww
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    
        confidences = []
        boxes = []
        classIds = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            # if confidence > self.confThreshold and detection[4] > self.objThreshold:
            if detection[4] > self.objThreshold:
                center_x = int((detection[0] - padw) * ratiow)
                center_y = int((detection[1] - padh) * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
            
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                classIds.append(classId)
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold).flatten()
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame
    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=4)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return frame
    def detect(self, srcimg):
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        img = self.preprocess(img)
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})
        
        # inference output
        det_outs = outs[0].squeeze(axis=0)
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.inpHeight/self.stride[i]), int(self.inpWidth/self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h,w):
                self.grid[i] = self._make_grid(w, h)

            det_outs[row_ind:row_ind+length, 0:2] = (det_outs[row_ind:row_ind+length, 0:2] * 2. - 0.5 + np.tile(self.grid[i],(self.na, 1))) * int(self.stride[i])
            det_outs[row_ind:row_ind+length, 2:4] = (det_outs[row_ind:row_ind+length, 2:4] * 2) ** 2 * np.repeat(self.anchor_grid[i],h*w, axis=0)
            row_ind += length
        srcimg = self.postprocess(srcimg, det_outs, padsize=(newh, neww, padh, padw))
        
        seg_outs = outs[1].squeeze(axis=0)
        mask = seg_outs[:, padh:(self.inpHeight - padh), padw:(self.inpWidth - padw)]
        seg_id = np.argmax(mask, axis=0).astype(np.uint8)
        seg_id = cv2.resize(seg_id, (srcimg.shape[1], srcimg.shape[0]), interpolation=cv2.INTER_NEAREST)
        # srcimg = np.array(Cityscapes_COLORMAP, dtype=np.uint8)[seg_id]
        # return srcimg
        
        seg_img = np.array(Cityscapes_COLORMAP, dtype=np.uint8)[seg_id]
        if srcimg.shape[0] < srcimg.shape[1]:
            return np.vstack((srcimg, seg_img))
        else:
            return np.hstack((srcimg, seg_img))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='images/berlin_000002_000019_leftImg8bit.png', help="image path")
    parser.add_argument('--modelpath', type=str, default='pspv5m_citybdd_conewaterbarrier.onnx')
    parser.add_argument('--confThreshold', default=0.5, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
    parser.add_argument('--objThreshold', default=0.5, type=float, help='object confidence')
    args = parser.parse_args()

    yolonet = yolov5(args.modelpath, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold, objThreshold=args.objThreshold)
    srcimg = cv2.imread(args.imgpath)
    srcimg = yolonet.detect(srcimg)

    winName = 'Deep learning object detection in ONNXRuntime'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()