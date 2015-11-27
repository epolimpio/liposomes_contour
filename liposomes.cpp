#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>
#include <algorithm>
#include <fstream>

using namespace std;
using namespace cv;

string type2str(int);

int main( int argc, char** argv )
{
	// Check if calling was OK
    if( argc != 2)
    {
     cout <<" Usage: ./liposomes [tifimage] [initial_frame] [final_frame]" << endl;
     return -1;
    }

    ofstream outfile;
    outfile.open("test.dat");

    // Read the file
    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_UNCHANGED);

    // Check for invalid input
    if(! image.data )                              
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }

    // Convert to 8-bit image using the whole dynamic range and display
    Mat image8bit;
    double min, max;
    normalize(image, image8bit, 255, 0, NORM_MINMAX, CV_8U);

    namedWindow( "8-bit original", CV_WINDOW_AUTOSIZE );
    imshow( "8-bit original", image8bit );

    // Blur the image to reduce noise (Gaussian Blur) and display
    Mat imageBlur;
    GaussianBlur(image8bit, imageBlur, Size(9, 9), 0, 0);

    namedWindow( "Blurred image", CV_WINDOW_AUTOSIZE );
    imshow( "Blurred image", imageBlur );

    // Get binary image with Otsu Threshold and display
    Mat imageBW;
    threshold(imageBlur, imageBW, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

    namedWindow( "Binary image", CV_WINDOW_AUTOSIZE );
    imshow( "Binary image", imageBW );

    // Closing of image to remove noise
    Mat imageMorph;
    Mat element = getStructuringElement(MORPH_RECT, Size(5,5), Point(-1,-1));
    morphologyEx(imageBW, imageMorph, MORPH_CLOSE, element);

    // Now we apply erosion to reduce noise further
    erode(imageMorph, imageMorph, element);

    // Display final binary image
    namedWindow( "Changed binary", CV_WINDOW_AUTOSIZE );
    imshow( "Changed binary", imageMorph );

    // Calculate distance transform to determine foreground
    Mat imageDist;
    distanceTransform(imageMorph, imageDist, CV_DIST_L2, 3);
	normalize(imageDist, imageDist, 0, 1., NORM_MINMAX);
    
    // Display distance image
    namedWindow( "Distance image", CV_WINDOW_AUTOSIZE );
    imshow( "Distance image", imageDist );

    // The biggest distances to zero are foreground 
    Mat imageDistBinary;
	threshold(imageDist, imageDistBinary, .7, 1., CV_THRESH_BINARY);
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    
    // Dilate to get a single contour
    dilate(imageDistBinary, imageDistBinary, kernel1);

    // Convert the threshold image to binary
	Mat imageDist8u;
    imageDistBinary.convertTo(imageDist8u, CV_8U);

    // Find foregroud marker contours and draw them
    vector<vector<Point> > contours;
    findContours(imageDist8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	Mat markers = Mat::zeros(imageDist.size(), CV_32SC1);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);

    // To get the background marker we invert the foreground binary
    // calculate distance and get the maximum location
    bitwise_not(imageDist8u, imageDist8u);
    distanceTransform(imageDist8u, imageDist, CV_DIST_L2, 3);
    Point2i min_loc, max_loc;
    minMaxLoc(imageDist, &min, &max, &min_loc, &max_loc);

    // Draw the background marker
    circle(markers, max_loc, 3, CV_RGB(255,255,255), -1);
    imshow("Markers", markers*10000);
    
    // Convert image to C3 to perform watershed and show the results
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
            else
                dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    }
    // Get the output to smooth edges
    cvtColor(dst, dst, CV_RGB2GRAY);
    Mat dst8bit;
	minMaxLoc(dst, &min, &max);
    normalize(dst, dst8bit, 255, 0, NORM_MINMAX, CV_8U);

    // Blur the edges and erode after threshold 
    GaussianBlur(dst8bit, dst8bit, Size(25, 25), 0, 0);
    normalize(dst8bit, dst8bit, 0, 1., NORM_MINMAX);
    threshold(dst8bit, dst8bit, .9, 1., CV_THRESH_BINARY );
    erode(dst8bit, dst8bit, kernel1, Point(-1,-1));

    // Find the contour of the mask
    vector<vector<Point> > contours2;
    findContours(dst8bit, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	
    // Define colors
	Scalar green(0,255,0);
	Scalar blue(255,0,0);
	Scalar red(0,0,255);

	// Parameters to be calculated to get geometrical properties
	double clen, carea, hullarea;
	double max_axis, min_axis, eqR;
	vector<vector<Point> >hull( contours2.size() );
	vector<RotatedRect> minEllipse( contours2.size() );

	// Calculate parameters for all the contours
    for (size_t i = 0; i < contours2.size(); i++){
    	// Calculated contour
        drawContours(imageC3, contours2, static_cast<int>(i), green, 2);

        // Convex Hull contour
        convexHull( Mat(contours2[i]), hull[i], false );
        drawContours(imageC3, hull, static_cast<int>(i), blue, 2);

        // Fit Ellipse to contour
        minEllipse[i] = fitEllipse( Mat(contours2[i]) );
        ellipse( imageC3, minEllipse[i], red, 2, 2 );

        // Perimeter, area and equivalent radius of liposome
    	clen = arcLength(contours2[i], true);
    	carea = contourArea(contours2[i]);
    	eqR = sqrt(carea/M_PI);

    	// Major ans minor axis
    	max_axis = std::max(minEllipse[i].size.height, minEllipse[i].size.width);
    	min_axis = std::min(minEllipse[i].size.height, minEllipse[i].size.width);

    	// Area of the convex Hull
    	hullarea = contourArea(hull[i]);

    	// Print parameters on screen
    	cout << "Equivalent Diameter: " << 2*eqR << endl;
    	cout << "Maj axis: " << max_axis << ", Min axis: " << min_axis << endl;
    	cout << "Elongation: " << log2(max_axis/min_axis) << endl;
    	cout << "Distortion: " << carea/clen/min_axis << endl;
    	cout << "Eccentricity: " << sqrt(1-pow(min_axis/max_axis, 2.0)) << endl;
    	cout << "Solidity: " << carea/hullarea << endl;
    	cout << "Roundness: " << clen/2/M_PI/eqR << endl;
    }

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