# EC463 Hardware Mini Project: Automated Vehicle Detection with a Raspberry Pi

Authors: McKenzie Cheung, Jessica Seto

   For the Hardware Mini Project, we opted for using the Open Computer Vision software for Python as its image and video processing capabilities are widely used, especially in the scientific community, for analysis. The pre-existing packages and libraries in Python 3.7 were also a large factor in our decision. The benefit of using already existing packages is that we have the ability to use pre-written functions to simplify our code. We chose our particular algorithm from an online repository because it contained rudimentary functions, such as multigrid size detection and color conversion, because these functions shortened the code while accomplishing the objective of the project. Although there are many repositories online that contain more complex algorithms, such as binarization through heat maps, we ultimately decided to use this particular algorithm because the project called for a short, simple, program with relatively accurate car detection abilities. In addition to the objective of the project, we also considered the capabilities of the Raspberry Pi and came to the conclusion that the Raspberry Pi might not be able to handle such long and complex algorithms, as its CPU does not possess the necessary processing capabilities. This is also a reason why Python is such a useful tool for a project such as this one. As Python is a scripting language, it takes much less time and processing power than a language that requires compiling, such as C++. For the limited power of the Raspberry Pi, this is especially key.
  
The algorithm we chose to use takes in the video and captures the frames from the clip. Using a training data file (.xml), the algorithm is able to use classifiers based on the .xml file, which has been trained with images of cars. 

```
   # Trained XML classifiers file for the cars     
   cars_class = cv2.CascadeClassifier('cars.xml'
```  

The algorithm then iterates over the frames in the video and converts them to grayscale. There are other methods that take this process and add another intermediary step, usually converting the image first into a binarized heatmap and then gray-scaling the image, but converting directly to grayscale is often the most efficient process. Based on the most promninent **white** shapes from the gray-scaled image, the algorithm is able to classify from the frame what is or is not a car.

```
   gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
```  

The algorithm would throw consistently throw a processing error once the video completely ended, so we added a “try and except” statement to mitigate the error and notify the user when the video had ended. 

``` 
   try: 
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)       
   except: 
        # When video ended, it would throw an error. This helps to overide the error once the video finishes running.         
        break
```
        
The algorithm used a detection function to determine cars of various sizes, based on the gray-scaling classification from the .xml file. We added a line of code to show the number of cars detected. From the gray-scaling, the image is processed as a matrix of RGB values, where **0** stands for black and **255** stands for white.

``` 
    # Detects cars of different sizes    
    car_detect = cars_class.detectMultiScale(gray, 1.1, 1)     
    print("This is the number of cars: ", len(car_detect))
```    

We also added a function to slow down the frames if the user wanted to see the video frame by frame in order to verify the accuracy of the number of cars counted.

```# Slows down the frames for better detection - Can be commented out if needed for faster processing  
 time.sleep(1.0)
```  
 
 The algorithm also draws a colored box over each detected car, which it determines again by which objects from the gray-scaled image are **white**. The **255** is the RGB code for white.
 
```
    # Draw rectangle over each car    
    num_cars = 0  
    for (x, y, w, h) in car_detect:  
        cv2.rectangle(frames, (x,y), (x+w, y+h), (0,0,255), 2)
```    
