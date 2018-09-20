# Senior_Design_Mini
# EC463 Hardware Mini Project: Automated Vehicle Detection with a Raspberry Pi

Authors: McKenzie Cheung, Jessica Seto

  For the Hardware Mini Project, we opted for using the Open Computer Vision software for Python as its image and video processing capabilities are widely used, especially in the scientific community, for analysis. The pre-existing packages and libraries in Python 3.7 were also a large factor in our decision. The benefit of using already existing packages is that we have the ability to use pre-written functions to simplify our code. We chose our particular algorithm from an online repository because it contained rudimentary functions, such as multigrid size detection and color conversion, because these functions shortened the code while accomplishing the objective of the project. Although there are many repositories online that contain more complex algorithms, such as binarization through heat maps, we ultimately decided to use this particular algorithm because the project called for a short, simple, program with relatively accurate car detection abilities. In addition to the objective of the project, we also considered the capabilities of the Raspberry Pi and came to the conclusion that the Raspberry Pi might not be able to handle such long and complex algorithms, as its CPU does not possess the necessary processing capabilities. This is also a reason why Python is such a useful tool for a project such as this one. As Python is a scripting language, it takes much less time and processing power than a language that requires compiling, such as C++. For the limited power of the Raspberry Pi, this is especially key.
  
  The way the algorithm we used works

  The algorithm we used was relatively accurate in detecting cars but it also detected a lot of other objects such as people and bicyclists, creating a lot of false positives. The system can definitely be improved in regards to accuracy. From observing the behavior of our system, we believe that one of the largest causes of the false detection was the video resolution produced by the Raspberry Pi’s camera. In some of our earlier video samples we took as test data, we filmed on a sunny day, which we think created a large margin of error for the algorithm. An excess of light can often wash out the resolution of an image or video, causing the frames to look faded or nearly white. 

 can have other detection methods such as keeping a heat map between consecutive frames and  using pipeline scanning.
	It isn’t a good idea to use user defined functions because limited the scope of data being analyzed may result in inaccurate calculations. Allowing the code to analyze the data  with changing parameters are more likely to produce more accurate results. User defined functions also hinder the performance of the code because it gives the optimizer insufficient and possibly misleading information. 
	
	

