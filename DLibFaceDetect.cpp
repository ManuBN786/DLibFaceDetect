/************************************************************************************************************************
 * Author     : Manu BN
 * Description: Detect faces using DLib and Haar Cascades and compare the results .

 ***********************************************************************************************************************/

//#include "opencv2/photo.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/opencv.h>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include "opencv2/objdetect/objdetect.hpp"

//#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>



using namespace cv;
using namespace dlib;
using namespace std;

cv::Rect dlib2cv(const dlib::rectangle& r);
int main(){

    image_window win;
    // Load face detection and pose estimation models.
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;
    deserialize("/home/user/Alignment/face/shape_predictor_68_face_landmarks.dat") >> pose_model;

          cv::Mat temp = imread("/home/user/Cartoon/comics-original.jpg");

           if(temp.empty()){

              cout<<"No iMage"<<endl;
               return -1;
           }


    // Detect Face using Haar Cascade

    CascadeClassifier face_cascade;
    face_cascade.load( "/home/user/OpenCV_Installed/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_alt2.xml" );
    std::vector<Rect> f;
    face_cascade.detectMultiScale(temp,f, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE);
    cv::rectangle(temp,f[0], Scalar(0,0,0), 1, 1);

    // Detect Face using DLib
            cv_image<rgb_pixel> cimg(temp);

    // Detect faces
               std::vector<dlib::rectangle> faces = detector(cimg);

   // cv::Rect R = dlib2cv(faces[0]); Convert Dlib rect to OpenCV rect
    cv::Rect R;
    R.x = faces[0].left();
    R.y = faces[0].top();
    R.width = faces[0].width();
    R.height = faces[0].height();


    cv::rectangle(temp,R, Scalar(255,255, 0), 1, 1);
    // Find the pose of each face.
    std::vector<full_object_detection> shapes;
    for (unsigned long i = 0; i < faces.size(); ++i)
        shapes.push_back(pose_model(cimg, faces[i]));

    // Display it all on the screen
    win.clear_overlay();
    win.set_image(cimg);
    win.add_overlay(render_face_detections(shapes));

    imshow("Image",temp);

    waitKey(0);
    return 0;
}

// Function to convert a DLib rectange to OpenCV rect
cv::Rect dlib2cv(const dlib::rectangle& r)
{
    return cv::Rect(r.left(), r.top(), r.width(), r.height());
}
