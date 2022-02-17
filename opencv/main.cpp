#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
};

const char *Cityscapes_Class[] = { "road", "sidewalk", "building", "wall", "fence",
"pole", "traffic light", "traffic sign", "vegetation",
"terrain", "sky", "person", "rider", "car", "truck",
"bus", "train", "motorcycle", "bicyle" };

const int Cityscapes_COLORMAP[19][3] = { {128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156},
{190, 153, 153}, {153, 153, 153}, {250, 170, 30}, {220, 220, 0}, {107, 142, 35}, {152, 251, 152},
{0, 130, 180}, {220, 20, 60}, {255, 0, 0}, {0, 0, 142}, {0, 0, 70}, {0, 60, 100}, {0, 80, 100}, {0, 0, 230}, {119, 11, 32} };

class YOLO
{
public:
	YOLO(Net_config config);
	Mat detect(Mat& frame);
private:
	const float anchors[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
								 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
								 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };
	const float stride[3] = { 8.0, 16.0, 32.0 };
	const int inpWidth = 1024;
	const int inpHeight = 1024;
	vector<string> class_names;
	int num_class;
	const int area = this->inpHeight*this->inpWidth;
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	const bool keep_ratio = true;
	Net net;
	void drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid);
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);
};

YOLO::YOLO(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;

	string modelFile = "pspv5m_citybdd_conewaterbarrier.onnx";
	this->net = readNet(modelFile);
	ifstream ifs("class.names");
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();
}

Mat YOLO::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLO::drawPred(float conf, int left, int top, int right, int bottom, Mat& frame, int classid)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	label = this->class_names[classid] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
}

Mat YOLO::detect(Mat& frame)
{
	Mat seg_img = frame.clone();
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
	Mat blob = blobFromImage(dstimg, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	int num_proposal = outs[1].size[1];
	int nout = outs[1].size[2];
	if (outs[1].dims > 2)
	{
		outs[1] = outs[1].reshape(0, num_proposal);
	}
	/////generate proposals
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> classIds;
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	int n = 0, q = 0, i = 0, j = 0, row_ind = 0; ///xmin,ymin,xamx,ymax,box_score,class_score
	float* pdata = (float*)outs[1].data;
	for (n = 0; n < 3; n++)   ///ÌØÕ÷Í¼³ß¶È
	{
		int num_grid_x = (int)(this->inpWidth / this->stride[n]);
		int num_grid_y = (int)(this->inpHeight / this->stride[n]);
		for (q = 0; q < 3; q++)    ///anchor
		{
			const float anchor_w = this->anchors[n][q * 2];
			const float anchor_h = this->anchors[n][q * 2 + 1];
			for (i = 0; i < num_grid_y; i++)
			{
				for (j = 0; j < num_grid_x; j++)
				{
					float box_score = pdata[4];
					if (box_score > this->objThreshold)
					{
						Mat scores = outs[1].row(row_ind).colRange(5, nout);
						Point classIdPoint;
						double max_class_socre;
						// Get the value and location of the maximum score
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						int class_idx = classIdPoint.x;
						//if (max_class_socre > this->confThreshold)
						//{ 
						float cx = (pdata[0] * 2.f - 0.5f + j) * this->stride[n];  ///cx
						float cy = (pdata[1] * 2.f - 0.5f + i) * this->stride[n];   ///cy
						float w = powf(pdata[2] * 2.f, 2.f) * anchor_w;   ///w
						float h = powf(pdata[3] * 2.f, 2.f) * anchor_h;  ///h

						int left = int((cx - padw - 0.5 * w)*ratiow);
						int top = int((cy - padh - 0.5 * h)*ratioh);

						confidences.push_back((float)max_class_socre);
						boxes.push_back(Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));
						classIds.push_back(class_idx);
						//}
					}
					row_ind++;
					pdata += nout;
				}
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame, classIds[idx]);
	}
	
	ratioh = (float)newh / frame.rows;
	ratiow = (float)neww / frame.cols;
	int seg_num_class = outs[0].size[1];
	float* pseg = (float*)outs[0].data;
	for (i = 0; i < seg_img.rows; i++)
	{
		for (j = 0; j < seg_img.cols; j++)
		{
			const int x = int(j*ratiow) + padw;
			const int y = int(i*ratioh) + padh;
			float max_prob = -1;
			int max_ind = 0;
			for (n = 0; n < seg_num_class; n++)
			{
				float pix_data = pseg[n * area + y * this->inpWidth + x];
				if (pix_data > max_prob)
				{
					max_prob = pix_data;
					max_ind = n;
				}
			}
			seg_img.at<Vec3b>(i, j)[0] = Cityscapes_COLORMAP[max_ind][0];
			seg_img.at<Vec3b>(i, j)[1] = Cityscapes_COLORMAP[max_ind][1];
			seg_img.at<Vec3b>(i, j)[2] = Cityscapes_COLORMAP[max_ind][2];
		}
	}

	Mat combine;
	if (frame.rows < frame.cols)
	{
		vconcat(frame, seg_img, combine);
	}
	else
	{
		hconcat(frame, seg_img, combine);
	}
	return combine;
}

int main()
{
	Net_config yolo_nets = { 0.3, 0.5, 0.3 };
	YOLO yolo_model(yolo_nets);
	string imgpath = "images/berlin_000002_000019_leftImg8bit.png";
	Mat srcimg = imread(imgpath);
	Mat outimg = yolo_model.detect(srcimg);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, outimg);
	waitKey(0);
	destroyAllWindows();
}