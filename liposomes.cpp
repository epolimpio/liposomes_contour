#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

string type2str(int);

int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }

    // Convert to 8-bit image using the whole dynamic range
    Mat image8bit;
    double min, max;
    cout << min << ", " << max << endl; 
	minMaxLoc(image, &min, &max);
    
    image.convertTo(image8bit, CV_8U, 255.0/(max-min), 255.0*min/(min-max));

    // Display 8-bit image
    namedWindow( "8-bit original", CV_WINDOW_AUTOSIZE );
    imshow( "8-bit original", image8bit );

    // Blur the image to reduce noise (Gaussian Blur)
    Mat imageBlur;
    GaussianBlur(image8bit, imageBlur, Size(9, 9), 0, 0);

    // Display blurred image
    namedWindow( "Blurred image", CV_WINDOW_AUTOSIZE );
    imshow( "Blurred image", imageBlur );

    // Get binary image with Otsu Threshold
    Mat imageBW;
    threshold(image8bit, imageBW, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    // Display binary image
    namedWindow( "Binary image", CV_WINDOW_AUTOSIZE );
    imshow( "Binary image", imageBW );

    // Opening image to remove noise
    Mat imageMorph;
    Mat element = getStructuringElement(MORPH_RECT, Size(5,5), Point(-1,-1));
    morphologyEx(imageBW, imageMorph, MORPH_CLOSE, element);

    // Now we apply dilation to determine foreground
    erode(imageMorph, imageMorph, element);

    // Display opening image
    namedWindow( "Changed binary", CV_WINDOW_AUTOSIZE );
    imshow( "Changed binary", imageMorph );

    // Calculate distance transform
    Mat imageDist;
    distanceTransform(imageMorph, imageDist, CV_DIST_L2, 3);
	minMaxLoc(imageDist, &min, &max);
	normalize(imageDist, imageDist, 0, 1., NORM_MINMAX);
    
    // Display distance image
    namedWindow( "Distance image", CV_WINDOW_AUTOSIZE );
    imshow( "Distance image", imageDist );

	threshold(imageDist, imageDist, .5, 1., CV_THRESH_BINARY);
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    dilate(imageDist, imageDist, kernel1);

	Mat imageDist8u;
    imageDist.convertTo(imageDist8u, CV_8U);

    // Find total markers
    vector<vector<Point> > contours;
    findContours(imageDist8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	Mat markers = Mat::zeros(imageDist.size(), CV_32SC1);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);

    // Draw the background marker
    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
    imshow("Markers", markers*10000);
    
    Mat imageC3;
    cvtColor(imageBlur, imageC3, CV_GRAY2RGB);

    // Perform the watershed algorithm
    watershed(imageC3, markers);

    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);

        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }

    // Create the result image
    Mat dst = Mat::zeros(markers.size(), CV_8UC3);

    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                dst.at<Vec3b>(i,j) = colors[index-1];
            else if (index == -1){
            	imageC3.at<Vec3b>(i,j) = Vec3b(0,0,255);
            }
            else
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    }
    cvtColor(dst, dst, CV_RGB2GRAY);
    Mat dst8bit;
	minMaxLoc(dst, &min, &max);
    
    dst.convertTo(dst8bit, CV_8U, 255.0/(max-min), 255.0*min/(min-max));
    GaussianBlur(dst8bit, dst8bit, Size(25, 25), 0, 0);
    normalize(dst8bit, dst8bit, 0, 1., NORM_MINMAX);
    threshold(dst8bit, dst8bit, .9, 1., CV_THRESH_BINARY );
    erode(dst8bit, dst8bit, kernel1, Point(-1,-1));
    vector<vector<Point> > contours2;
    findContours(dst8bit, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	Scalar color(0,255,0);
    for (size_t i = 0; i < contours2.size(); i++)
        drawContours(imageC3, contours2, static_cast<int>(i), color, 1);

    // Visualize the final image
    imshow("Final Result", dst);

    // Display 8-bit image
    namedWindow( "Image with contour", CV_WINDOW_AUTOSIZE );
    imshow( "Image with contour", imageC3 );

    waitKey(0); // Wait for a keystroke in the window
    return 0;
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}