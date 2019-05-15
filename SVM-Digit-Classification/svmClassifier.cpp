#include <QCoreApplication>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;
using namespace cv::ml;

const char *windowName = "SVM Example";
Ptr<SVM> svm;
bool Clicked;
Mat srcImg;

void predictDigit(){

    Mat gray;
   // String imgName = format("data/test/%d.png",2);
   // srcImg = imread(imgName,1);
    cvtColor(srcImg,gray,COLOR_BGR2GRAY);
    Mat tmp1, tmp2;
    resize(gray,tmp1, Size(72,72), 0,0,INTER_LINEAR );
    tmp1.convertTo(tmp2,CV_32FC1);

    float prediction = svm->predict(tmp2.reshape(1,1));

    Mat out = Mat(240,240, CV_8UC3, Scalar(255,255,255));
    if(prediction == 1){
       putText(out,"1", Point(70, 190), FONT_HERSHEY_PLAIN, 14, CV_RGB(0,200,0), 15.8);
    }
    else if(prediction == 0){
        putText(out,"0", Point(70, 190), FONT_HERSHEY_PLAIN, 14, CV_RGB(0,200,0), 15.8);
    }

    imshow("Prediction", out);

}
void onMouseAction(int event, int x, int y, int f, void *) {

  switch (event) {

  case EVENT_LBUTTONDOWN:
      Clicked= true;
      srcImg = Mat(240,240, CV_8UC3, Scalar(255,255,255));
      break;

  case EVENT_LBUTTONUP:
    predictDigit();
    Clicked = false;
    break;

  case EVENT_MOUSEMOVE:
    if (Clicked) {
        circle(srcImg,Point(x,y),10,Scalar(0,0,0),-1);
        imshow(windowName, srcImg);

    }
    break;

  default:
    break;
  }




}

int main()
{
   Mat trainingDataMat;
   Mat label_array;
   String imgName;

   for(int i=1;i<11;i++){

       //Create one data and label
       imgName = format("data/one/%d.png",i);
       Mat src = imread(imgName,0);
       Mat tmp1, tmp2;
       resize(src,tmp1, Size(72,72), 0,0,INTER_LINEAR );
       tmp1.convertTo(tmp2,CV_32FC1);
       trainingDataMat.push_back(tmp2.reshape(1,1));
       label_array.push_back(1);

       //Create zero data and label
       imgName = format("data/zero/%d.png",i);
       src = imread(imgName,0);
       resize(src,tmp1, Size(72,72), 0,0,INTER_LINEAR );
       tmp1.convertTo(tmp2,CV_32FC1);
       trainingDataMat.push_back(tmp2.reshape(1,1));
       label_array.push_back(0);

   }

    Mat labelsMat;
    labelsMat=label_array.reshape(1,1); //make continuous



       //Create the SVM
       svm = SVM::create();

       // Set up SVM's parameters
       svm->setType(ml::SVM::C_SVC);
       svm->setKernel(ml::SVM::LINEAR);
       //svm->setGamma(3);
       svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

       Ptr<ml::TrainData> tData = ml::TrainData::create(trainingDataMat, ml::SampleTypes::ROW_SAMPLE, labelsMat);
       svm->train(tData);
       svm->train( trainingDataMat , ml::ROW_SAMPLE , labelsMat );

       srcImg = Mat(240,240, CV_8UC3, Scalar(255,255,255));
       namedWindow(windowName, 1);
       setMouseCallback(windowName, onMouseAction, NULL);
       imshow(windowName, srcImg);



       while (1) {
         char c = waitKey();
         if (c == 27)
           break;

       }
 

}
