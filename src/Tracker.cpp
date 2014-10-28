#include "Tracker.h"

Tracker::Tracker()
{
	
}

Tracker::~Tracker()
{
	delete mBlobDetector;
	mImage.reset();
}

void Tracker::setup()
{
	try {
		mCapture = Capture(getWindowWidth(), getWindowHeight());
		mCapture.start();
	}
	catch (...) {
		console() << "Failed to initialize capture" << std::endl;
	}

	mApproxEps = 1.0;
	mCannyThresh = 200;

	mPickedColor = Color8u(255.f * 0.541176f, 255.f * 0.427451f, 255.f * 0.564706f);
	setTrackingHsv();

	mScaling = 2.5f;

	//this is the 2nd algorithm
	cv::SimpleBlobDetector::Params params;
	params.minThreshold = 5;
	params.maxThreshold = 255;
	params.thresholdStep = 5;

	params.minArea = 2;
	params.minConvexity = 0.3;
	params.minInertiaRatio = 0.01;

	params.maxArea = 500;
	params.maxConvexity = 10;

	params.filterByColor = false;
	params.filterByCircularity = false;
	
	mBlobDetector = new cv::SimpleBlobDetector(params);
	mBlobDetector->create("SimpleBlob");
}

void Tracker::setTrackingHsv()
{
	Color8u col = Color(mPickedColor);
	Vec3f colorHSV = col.get(CM_HSV);
	colorHSV.x *= 179;
	colorHSV.y *= 255;
	colorHSV.z *= 255;

	mColorMin = cv::Scalar(colorHSV.x - 5, colorHSV.y - 50, colorHSV.z - 50);
	mColorMax = cv::Scalar(colorHSV.x + 5, 255, 255);

}

void Tracker::mouseDown(Vec2i& mousePos)
{
	if (mImage && mImage.getBounds().contains(mousePos)) {

		mPickedColor = mImage.getPixel(mousePos);
		//console() << "r: " << mPickedColor.r << " g:" << mPickedColor.g << " b: " << mPickedColor.b << std::endl;
		setTrackingHsv();
	}
}

void Tracker::update()
{
	if (mCapture && mCapture.checkNewFrame()) {
		mImage = mCapture.getSurface();
		mCaptureTex = gl::Texture(mImage);

		// do some CV
		cv::Mat inputMat(toOcv(mImage));

		cv::resize(inputMat, inputMat, cv::Size(getWindowWidth()/mScaling, getWindowHeight()/mScaling));

		cv::Mat inputHSVMat, frameTresh;
		cv::cvtColor(inputMat, inputHSVMat, CV_BGR2HSV);

		cv::inRange(inputHSVMat, mColorMin, mColorMax, frameTresh);

		cv::medianBlur(frameTresh, frameTresh, 7);

		cv::Mat cannyMat;
		cv::Canny(frameTresh, cannyMat, mCannyThresh, mCannyThresh*2.f, 3);

		vector< std::vector<cv::Point> >  contours;
		cv::findContours(cannyMat, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

		mCenters = vector<cv::Point2f>(contours.size());
		mRadius = vector<float>(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			std::vector<cv::Point> approxCurve;
			cv::approxPolyDP(contours[i], approxCurve, mApproxEps, true);
			cv::minEnclosingCircle(approxCurve, mCenters[i], mRadius[i]);
		}
	}
}

void Tracker::update2()
{
	if (mCapture && mCapture.checkNewFrame()) {
		mImage = mCapture.getSurface();
		mCaptureTex = gl::Texture(mImage);

		cv::Mat inputMat(toOcv(mImage));

		cv::resize(inputMat, inputMat, cv::Size(getWindowWidth() / mScaling, getWindowHeight() / mScaling));

		vector< cv::KeyPoint >  keyPoints;
		mBlobDetector->detect(inputMat, keyPoints);

		mCenters = vector<cv::Point2f>(keyPoints.size());
		mRadius = vector<float>(keyPoints.size());
		for (int i = 0; i < keyPoints.size(); i++)
		{
			mCenters.push_back(keyPoints[i].pt);
			mRadius.push_back(keyPoints[i].size);
		}
	}
	else
	{
		console() << "WARNING: skipped a frame";
	}
}

void Tracker::draw()
{
	if (mCaptureTex) {
		gl::draw(mCaptureTex);

		gl::color(Color::white());

		for (int i = 0; i < mCenters.size(); i++)
		{
			Vec2f center = fromOcv(mCenters[i])*mScaling;
			gl::begin(GL_POINTS);
			gl::vertex(center);
			gl::end();

			//gl::drawStrokedCircle(center, mRadius[i] * mScaling);
		}
	}
}