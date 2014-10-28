#pragma once

#include "cinder/Vector.h"
#include "cinder/app/AppNative.h"

#include "cinder/gl/gl.h"
#include "cinder/gl/Texture.h"
#include "cinder/Surface.h"
#include "cinder/ImageIo.h"
#include "cinder/cairo/Cairo.h"
#include "cinder/Capture.h"
#include "CinderOpenCV.h"


using namespace ci;
using namespace ci::app;
using namespace std;

class Tracker
{
public:
	Tracker();
	~Tracker();
	void setTrackingHsv();
	void setup();
	void update();
	void update2();
	void draw();
	void mouseDown(Vec2i &mousePos);

	Surface8u   mImage;
	double mApproxEps;
	int mCannyThresh;
	Capture     mCapture;
	gl::Texture mCaptureTex;
	ColorA      mPickedColor;
	cv::Scalar  mColorMin;
	cv::Scalar  mColorMax;
	vector<cv::Point2f> mCenters;
	vector<float> mRadius;
	int mMaxCenters;
	float mScaling; //the scale difference between the mCenters and actual screen
	cv::SimpleBlobDetector *mBlobDetector;

};