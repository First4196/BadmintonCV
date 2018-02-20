#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int WIDTH = 500;
int HEIGHT = 500;
int MIN_GAP = 120;

Mat kernel = getStructuringElement(MORPH_ELLIPSE,Size(3,3),Point(1,1));

Mat bgr, ycrcb, whiteMask, lineImg, bestCourtImg, yellow, bgImg, fgImg, fgColorImg;
vector<Vec4i> lines, Wlines, Elines, Nlines, Slines;

Point2f cNW(0,0), cNE(200,0), cSW(0,440), cSE(200,440);
vector<Point2f> courtCorners({cNW,cNE,cSW,cSE});
vector<Vec4f> courtLines({
    // hlines
    {0,0,200,0},
    {0,25,200,25},
    {0,155,200,155},
    {0,285,200,285},
    {0,415,200,415},
    {0,440,200,440},
    // vlines
    {0,0,0,440},
    {15,0,15,440},
    {100,0,100,155},
    {100,285,100,440},
    {185,0,185,440},
    {200,0,200,440}
});
Vec4f middleCourtLine({0,220,200,220});

Rect boundingBoxN, boundingBoxS;
Point2f transformedFeetN, transformedFeetS;

double bestScore = -INFINITY;
Point2f bestNW, bestNE, bestSW, bestSE;
Rect bestCourtROI;

Point3f intersection(const Vec4f &l1, const Vec4f &l2){
    
    Point2f p1(l1[0], l1[1]), q1(l1[2], l1[3]);
    Point2f p2(l2[0], l2[1]), q2(l2[2], l2[3]);

    double a1 = q1.y-p1.y;
    double b1 = p1.x-q1.x;
    double c1 = a1*p1.x + b1*p1.y;
    
    double a2 = q2.y-p2.y;
    double b2 = p2.x-q2.x;
    double c2 = a2*p2.x + b2*p2.y;

    double det = a1*b2 - a2*b1;
    if(det != 0){
        double x = (b2*c1 - b1*c2)/det;
        double y = (a1*c2 - a2*c1)/det;
        return Point3f(x,y,1);
    }
    else{
        return Point3f(0,0,-1);
    }
    
}

void drawWhiteLine(Mat &mat, Vec4f l){
    Point2f p(l[0], l[1]), q(l[2], l[3]);
    line(mat, p, q, Scalar(255,255,255), 1, CV_AA);
}

bool isVertical(const Vec4i &l){
    Point2f p(l[0], l[1]), q(l[2], l[3]);
    return (p.x-q.x)*(p.x-q.x) < (p.y-q.y)*(p.y-q.y);
}

void showImage(const char* name, const Mat &img, bool wait = true){
    namedWindow(name, WINDOW_AUTOSIZE );
    imshow(name, img);
    waitKey(20);
    if(wait) waitKey(0);
}

void preProcess(){
    resize(bgr, bgr, Size(WIDTH,HEIGHT));
    cvtColor(bgr, ycrcb, CV_BGR2YCrCb);
    yellow = Mat::zeros( bgr.size(), CV_8UC3);
    yellow = Scalar(0,255,255);
    whiteMask = Mat::zeros( bgr.size(), CV_8U );
    int t1=6, t2=10, sl=128, sd=20;
    for(int x=t2; x<WIDTH-t2; x++){
        for(int y=t2; y<HEIGHT-t2; y++){
            bool isWhite = false;
            uchar here = ycrcb.at<Vec3b>(y,x)[0];
            if(here>sl){
                for(int t=t1; t<=t2; t++){
                    isWhite |= here-ycrcb.at<Vec3b>(y,x-t)[0]>sd && here-ycrcb.at<Vec3b>(y,x+t)[0]>sd;
                    isWhite |= here-ycrcb.at<Vec3b>(y-t,x)[0]>sd && here-ycrcb.at<Vec3b>(y+t,x)[0]>sd;
                }
            }
            if(isWhite){
                whiteMask.at<uchar>(y,x) = 255;
            }
        }
    }
    dilate(whiteMask, whiteMask, kernel);
    //showImage("white",whiteMask,false);
}

void detectLines(){
    lineImg = Mat::zeros( bgr.size(), CV_8U);
    HoughLinesP(whiteMask, lines, 1, CV_PI/180, HEIGHT/6, HEIGHT/6, 10 );
    for(auto &l : lines){
        if(isVertical(l)){
            if(l[0]+l[2]<=WIDTH){
                Wlines.push_back(l);
            }
            else{
                Elines.push_back(l);
            }
        }
        else{
            if(l[1]+l[3]<=HEIGHT){
                Nlines.push_back(l);
            }
            else{
                Slines.push_back(l);
            }
        }
        drawWhiteLine(lineImg,l);
    }
    //showImage("line",lineImg,false);
}

tuple<vector<Vec4f>,Vec4f> getTransformedCourtLines(Point2f NW, Point2f NE, Point2f SW, Point2f SE){
    
    vector<Point2f> intersectionCorners({NW,NE,SW,SE});
    Mat homoMat = getPerspectiveTransform(courtCorners, intersectionCorners);
    
    vector<Point2f> courtPoints1, courtPoints2;
    courtLines.push_back(middleCourtLine); 
    for(auto &l : courtLines){
        courtPoints1.push_back({l[0],l[1]});
        courtPoints2.push_back({l[2],l[3]});
    }
    courtLines.pop_back();

    vector<Point2f> transformedCourtPoints1, transformedCourtPoints2;
    perspectiveTransform(courtPoints1, transformedCourtPoints1, homoMat);
    perspectiveTransform(courtPoints2, transformedCourtPoints2, homoMat);
    
    vector<Vec4f> transformedCourtlines;
    for(size_t i=0; i<transformedCourtPoints1.size(); i++){
        transformedCourtlines.push_back({transformedCourtPoints1[i].x,transformedCourtPoints1[i].y,transformedCourtPoints2[i].x,transformedCourtPoints2[i].y});
    }

    Vec4f transformedMiddleCourtLine = transformedCourtlines.back();
    transformedCourtlines.pop_back();

    return {transformedCourtlines, transformedMiddleCourtLine};

}

void calculateOptimalCourt(){
    bestScore = -INFINITY;
    bestNW = {150,100};
    bestNE = {350,100};
    bestSW = {100,400};
    bestSE = {400,400};
    bestCourtImg = Mat::zeros( bgr.size(), CV_8U);
    int x = 1;
    for(auto &W : Wlines){
        cout << "Court : " << x << "/" << Wlines.size() << " " << bestScore << endl; x++;
        for(auto &E : Elines){
            int vGap = (E[0]+E[2])/2 - (W[0]+W[2])/2;
            if(vGap < MIN_GAP){
                continue;
            }
            for(auto &N : Nlines){
                for(auto &S : Slines){
                    int hGap = (S[1]+S[3])/2 - (N[1]+N[3])/2;
                    if(hGap < MIN_GAP){
                        continue;
                    }
                    Point3f tNW, tNE, tSW, tSE;
                    tNW = intersection(N,W);
                    tNE = intersection(N,E);
                    tSW = intersection(S,W);
                    tSE = intersection(S,E);
                    if(tNW.z<0 || tNE.z<0 || tSW.z<0 || tSE.z<0){
                        continue;
                    }
                    Point2f NW(tNW.x,tNW.y), NE(tNE.x,tNE.y), SW(tSW.x,tSW.y), SE(tSE.x,tSE.y);
                    vector<Vec4f> transformedCourtlines;
                    Vec4f transformedMiddleCourtLine;
                    tie(transformedCourtlines, transformedMiddleCourtLine) = getTransformedCourtLines(NW,NE,SW,SE);
                    double score = 0;
                    for(auto &l : transformedCourtlines){
                        for(double r=0; r<=1.0; r+=0.01){
                            int x = l[0]*r + l[2]*(1.0-r);
                            int y = l[1]*r + l[3]*(1.0-r);
                            if(x>=0 && x<WIDTH && y>=0 && y<HEIGHT){
                                score += whiteMask.at<uchar>(y, x);
                            }
                        }
                    }
                    if(score>bestScore){
                        bestScore = score;
                        bestNW = NW;
                        bestNE = NE;
                        bestSW = SW;
                        bestSE = SE;
                        Mat courtImg = Mat::zeros( bgr.size(), CV_8U);
                        for(auto &l : transformedCourtlines){
                            drawWhiteLine(courtImg,l);
                        }
                        drawWhiteLine(courtImg,transformedMiddleCourtLine);
                        bestCourtImg = courtImg;
                    } 
                }
            }
        }
    }
    bestCourtROI.x = min(bestNW.x,bestSW.x);
    bestCourtROI.y = min(bestNW.y,bestNE.y);
    bestCourtROI.width = max(bestNE.x,bestSE.x) - bestCourtROI.x;
    bestCourtROI.height = max(bestSW.y,bestSE.y) - bestCourtROI.y;
    //showImage("court", bestCourtImg, false);
}

void showCourt(){
    Mat result = bgr;
    yellow.copyTo(result, bestCourtImg);
    showImage("result",result);
}

void clearMem(){
    Wlines.clear();
    Elines.clear();
    Nlines.clear();
    Slines.clear();
}

void processImage(const char* path){
    bgr = imread( path, CV_LOAD_IMAGE_COLOR );
    if ( !bgr.data ){
        printf("No image data \n");
        return;
    }
    //showImage("bgr",bgr,false);
    preProcess();
    detectLines();
    calculateOptimalCourt();
    showCourt();
    clearMem();
}

int bucket[64][64][64];
void calculateBgImg(const char* path, int lo, int hi){
    VideoCapture capture(path);
    Mat frame;
    vector<Mat> frames;
    if( !capture.isOpened() ){
        printf("No video data \n");
        return;
    }
    //namedWindow(path, WINDOW_AUTOSIZE);
    for(int i=0;; i++){
        capture >> frame;
        if(frame.empty()){
            break;
        }
        if(i%50==0 && i>=lo && i<hi){
            cout << "Frame " << i << endl;
            resize(frame, frame, Size(WIDTH,HEIGHT));
            frames.push_back(frame);
            //imshow(path, frame);
            //waitKey(20);
        }
    }
    cout << "Frames : " << frames.size() << endl;
    bgImg = Mat::zeros( Size(WIDTH,HEIGHT), CV_8UC3);
    for(int x=0; x<WIDTH; x++){
        cout << "bgImg : " << x+1 << "/" << WIDTH << endl;
        for(int y=0; y<HEIGHT; y++){
            int mx=0;
            Vec3b mxc({0,0,0});
            for(int i=0; i<frames.size(); i++){
                Vec3b p = frames[i].at<Vec3b>(y,x);
                bucket[p[0]/4][p[1]/4][p[2]/4]++;
                if(bucket[p[0]/4][p[1]/4][p[2]/4]>mx){
                    mx = bucket[p[0]/4][p[1]/4][p[2]/4];
                    mxc = p;
                }
            }
            for(int i=0; i<frames.size(); i++){
                Vec3b p = frames[i].at<Vec3b>(y,x);
                bucket[p[0]/4][p[1]/4][p[2]/4]--;
            }
            bgImg.at<Vec3b>(y,x) = mxc;
        }
    }
    //showImage(path,bgImg,false);
}

double distance(Point2f p1, Point2f p2){
    return 10*(p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y);
}

Point2f getMean(vector<Point2f> &vec){
    // Might have problem with empty clusters
    assert(vec.size()>0);
    double sumX = 0, sumY = 0;
    for(Point2f u : vec){
        sumX += u.x;
        sumY += u.y;
    }
    return Point2f(sumX/vec.size(), sumY/vec.size());
}

tuple<Point2f,Point2f> twoMeans(Mat &fgImg){
    Point2f centerN(WIDTH/2, HEIGHT/5), centerS(WIDTH/2, HEIGHT*4/5); 
    vector<Point2f> points;
    for(int x=0; x<WIDTH; x++){
        for(int y=0; y<HEIGHT; y++){
            if(fgImg.at<uchar>(y,x) == 255){
                points.push_back(Point2f(x,y));
            }
        }
    }
    for(int i=0; i<100; i++){
        vector<Point2f> pointsCenterN, pointsCenterS;
        for(auto point : points){
            if(distance(point, centerN)<=distance(point, centerS)){
                pointsCenterN.push_back(point);
            }
            else{
                pointsCenterS.push_back(point);
            }
        }
        Point2f newCenterN = getMean(pointsCenterN);
        Point2f newCenterS = getMean(pointsCenterS);
        double changeN = distance(centerN, newCenterN);
        double changeS = distance(centerS, newCenterS);
        centerN = newCenterN;
        centerS = newCenterS;
        cout << "Iteration " << i << " : (" << centerN.x << "," << centerN.y << ") (" << centerS.x << "," << centerS.y << ")" << endl;
        if(changeN<1 && changeS<1){
            break;
        }
    }
    return {centerN, centerS};
}

Rect getBoundingBox(Point2f center, vector<Point2f> &points){
    assert(points.size()>1);

    double meanX = 0, meanY = 0;
    for(Point2f point : points){
        meanX += point.x;
        meanY += point.y;
    }
    meanX /= points.size();
    meanY /= points.size();

    double sdX = 0, sdY;
    for(Point2f point : points){
        sdX += (point.x-meanX)*(point.x-meanX);
        sdY += (point.y-meanY)*(point.y-meanY);
    }
    sdX = sqrt(sdX/(points.size()-1));
    sdY = sqrt(sdY/(points.size()-1));

    float loX=WIDTH-1, hiX=0, loY=HEIGHT-1, hiY=0;
    for(Point2f point : points){
        bool goodX = (meanX-3*sdX < point.x && point.x < meanX+3*sdX);
        bool goodY = (meanY-3*sdY < point.y && point.y < meanY+3*sdY);
        if(goodX && goodY){
            loX = min(loX, point.x);
            hiX = max(hiX, point.x);
            loY = min(loY, point.y);
            hiY = max(hiY, point.y);
        }    
    }
    return Rect( Point2f(loX, loY), Point2f(hiX, hiY) );
}

void showVideo(const char* path, int lo=0, int hi=2e9){

    ofstream outfile("detected.txt");

    VideoCapture capture(path);
    Mat frame;
    vector<Mat> frames;
    if( !capture.isOpened() ){
        printf("No video data \n");
        return;
    }
    namedWindow(path, WINDOW_AUTOSIZE);
    int splitY = -1;
    for(int i=0;; i++){
        capture >> frame;
        if(frame.empty()){
            break;
        }
        if(i>=lo && i<hi){
            resize(frame,frame,Size(WIDTH,HEIGHT));
            
            fgImg = Mat::zeros( Size(WIDTH,HEIGHT), CV_8U);
            int extend = bestCourtROI.y - max(0,bestCourtROI.y-100);
            Rect extendedBestCourtROI = bestCourtROI;
            extendedBestCourtROI.y = bestCourtROI.y-extend;
            extendedBestCourtROI.height = bestCourtROI.height+extend;
            
            Mat frameYCrCb, bgImgYCrCb;
            cvtColor(frame, frameYCrCb, CV_BGR2YCrCb);
            cvtColor(bgImg, bgImgYCrCb, CV_BGR2YCrCb);
            for(int i=0; i<extendedBestCourtROI.width; i++){
                for(int j=0; j<extendedBestCourtROI.height; j++){
                    int x = i+extendedBestCourtROI.x;
                    int y = j+extendedBestCourtROI.y;
                    Vec3b framePixel = frameYCrCb.at<Vec3b>(y,x);
                    Vec3b bgImgPixel = bgImgYCrCb.at<Vec3b>(y,x);
                    double dist = 0;
                    dist += (framePixel[1]-bgImgPixel[1])*(framePixel[1]-bgImgPixel[1]);
                    dist += (framePixel[2]-bgImgPixel[2])*(framePixel[2]-bgImgPixel[2]);
                    if(dist>100){
                        fgImg.at<uchar>(y,x) = 255;
                    }
                }
            }

            erode(fgImg,fgImg,kernel);
            erode(fgImg,fgImg,kernel);
            Point2f centerN, centerS;
            tie(centerN,centerS) = twoMeans(fgImg);

            fgColorImg = Mat::zeros( Size(WIDTH, HEIGHT), CV_8UC3);

            vector<Point2f> pointCenterN, pointCenterS;
            for(int x=0; x<WIDTH; x++){
                for(int y=0; y<HEIGHT; y++){
                    if(fgImg.at<uchar>(y,x) == 255){
                        if(distance(Point2f(x,y), centerN)<=distance(Point2f(x,y), centerS)){
                            pointCenterN.push_back(Point2f(x,y));
                        }
                        else{
                            pointCenterS.push_back(Point2f(x,y));
                        }
                    }
                }
            }

            boundingBoxN = getBoundingBox(centerN,pointCenterN);
            boundingBoxS = getBoundingBox(centerS,pointCenterS);

            Point2f feetN(centerN.x, centerN.y*0.2+boundingBoxN.br().y*0.8);
            Point2f feetS(centerS.x, centerS.y*0.2+boundingBoxS.br().y*0.8);

            for(Point2f point : pointCenterN){
                circle(fgColorImg, point, 1, Scalar(0,255,255), 1);
            }
            rectangle(fgColorImg, boundingBoxN, Scalar(255,0,0), 2);
            //circle(fgColorImg, centerN, 2, Scalar(255,0,0), 2);
            circle(fgColorImg, feetN, 2, Scalar(255,0,0), 2);
            
            for(Point2f point : pointCenterS){
                circle(fgColorImg, point, 1, Scalar(255,0,255), 1);
            }
            rectangle(fgColorImg, boundingBoxS, Scalar(255,0,0), 2);
            //circle(fgColorImg, centerS, 2, Scalar(255,0,0), 2);
            circle(fgColorImg, feetS, 2, Scalar(255,0,0), 2);
            
            //imshow("fgColorImg", fgColorImg);
            
            //rectangle(frame, extendedBestCourtROI.tl(), extendedBestCourtROI.br(), Scalar(0,0,255), 2);
            yellow.copyTo(frame,bestCourtImg);
            
            rectangle(frame, boundingBoxN, Scalar(255,0,0), 2);
            circle(frame, feetN, 2, Scalar(255,0,0), 2);
            
            rectangle(frame, boundingBoxS, Scalar(255,0,0), 2);
            circle(frame, feetS, 2, Scalar(255,0,0), 2);

            imshow(path, frame);

            vector<Point2f> bestCorners({bestNW,bestNE,bestSW,bestSE});
            Mat homoMat = getPerspectiveTransform(bestCorners, courtCorners);
            Mat topView = Mat::zeros(Size(300,540), CV_8UC3);
            vector<Point2f> points({feetN, feetS}), transformedPoints;
            perspectiveTransform(points, transformedPoints, homoMat);
            for(auto &l : courtLines){
                drawWhiteLine(topView, Vec4f(l[0]+50,l[1]+50,l[2]+50,l[3]+50));
            }
            Vec4f l = middleCourtLine;
            drawWhiteLine(topView, Vec4f(l[0]+50,l[1]+50,l[2]+50,l[3]+50));
            transformedFeetN = transformedPoints[0];
            transformedFeetS = transformedPoints[1];
            circle(topView, Point2f(transformedFeetN.x+50, transformedFeetN.y+50), 2, Scalar(255,0,0), 2);
            circle(topView, Point2f(transformedFeetS.x+50, transformedFeetS.y+50), 2, Scalar(255,0,0), 2);
            imshow("TopView", topView);
            
            string s = to_string(i) + " ";
            
            s += to_string(boundingBoxN.x) + " ";
            s += to_string(boundingBoxN.y) + " ";
            s += to_string(boundingBoxN.width) + " ";
            s += to_string(boundingBoxN.height) + " ";

            s += to_string(boundingBoxS.x) + " ";
            s += to_string(boundingBoxS.y) + " ";
            s += to_string(boundingBoxS.width) + " ";
            s += to_string(boundingBoxS.height) + " ";

            s += to_string(transformedFeetN.x) + " ";
            s += to_string(transformedFeetN.y) + " ";

            s += to_string(transformedFeetS.x) + " ";
            s += to_string(transformedFeetS.y) + "\n";

            outfile << s;

            waitKey(20);
        }
    }

    outfile.close();

}

void processVideo(const char* path, int lo=0, int hi=2e9){
    calculateBgImg(path,lo,hi);
    bgr = bgImg;
    //showImage("bgr",bgr,false);
    
    preProcess();
    detectLines();
    calculateOptimalCourt();
    //showCourt();
    clearMem();
    
    showVideo(path,lo,hi);
}

int main(int argc, char** argv ){
    // processImage("court1.jpg");
    // processImage("court2.jpg");
    // processImage("court3.jpg");
    // processImage("court4.jpg");
    // processImage("court5.jpg");
    processVideo("video1.mp4", 200, 3200);
    return 0;
}