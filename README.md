# Attendance_monitor_by_face_recog

The basic idea behind this project is counting daily attendance of students by recognize their faces and prepare csv that store day to day attendance count so it is easy to maintain . 
For this I use deep learning Inception_resnet model which will trained facial features of each students and return 128D vector that contain unique features of face and also use landmarks of eye , nose , mouth so it is more uniquely identify faces from others. To avoid problem like spoof face detection by using image of that person/students I add eye blink detection that will confirmed that in-front of camera real person/students is there if someone try by using image of absent student then it will not recognize them.
