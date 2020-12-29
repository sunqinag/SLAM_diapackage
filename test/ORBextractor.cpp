//
// Created by 孙强 on 2020/12/29.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include "../include/ORBextractor.hpp"
#include <iomanip>

using namespace std;
using namespace cv;

// 每一帧提取的特征点数 1000
int nFeatures = 1000;
// 图像建立金字塔时的变化尺度 1.2
float fScaleFactor = 1.2;
// 尺度金字塔的层数 8
int nLevels = 8;
// 提取fast特征点的默认阈值 20
int fIniThFAST = 20;
// 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
int fMinThFAST = 8;

void drewKeypointInImage(Mat img, vector<KeyPoint> keypoints) {
    for (auto &point:keypoints) {
        circle(img, point.pt, 2, Scalar(0, 255, 0));
    }
}


int main() {
    ORBextractor *extractor = new ORBextractor(nFeatures,      //参数的含义还是看上面的注释吧
                                               fScaleFactor,
                                               nLevels,
                                               fIniThFAST,
                                               fMinThFAST);

    Mat mask;
    vector<KeyPoint> keypoints;
    Mat descriptors;

    int count = 0;
    while (1) {
        string img_dir = "/Users/sunqiang/data/dataset/sequences/00/image_0/";
        stringstream ss;
        ss << setfill('0') << setw(6) << count;
        string img_path = img_dir + ss.str() + ".png";

        Mat img = imread(img_path, 0);
        extractor->extract_features(img, mask, keypoints, descriptors);
        cout << "keypoints size:" << keypoints.size() << endl;

        Mat img_3channel;
        cvtColor(img, img_3channel, cv::COLOR_GRAY2RGB);
        drewKeypointInImage(img_3channel, keypoints);
        imshow("image", img_3channel);
        waitKey(1);
        count++;
    }


    return 0;
}