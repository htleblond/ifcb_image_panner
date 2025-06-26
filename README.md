# ifcb_image_panner

I am not affiliated with the Woods Hole Oceanographic Institution, nor was I involved in the development of the Imaging FlowCytobot. This is just some code for pre-processing IFCB images before being input into the ifcb_classifier (https://github.com/WHOIGit/ifcb_classifier), which I was also not involved with.

Basically, this is a tool that preserves the size of plankton in IFCB images by either panning to a part of the image when one (or both) of the dimensions is longer than 299 (or whatever size is specified) pixels, and/or adding stochastic padding to simulate the background when one (or both) of the dimensions is smaller than 299 pixels. The ifcb_classifier instead simply resizes the images, which changes the size of the objects in the image and distorts the aspect ratio, so this is meant to remedy that.

To use this program, simply download the .py file from the latest release and have SciPy, Pillow and NumPy (version 1.26.4) installed via pip, which can easily be done via the attached batch file. For instructions, input `python ifcb_image_panner.py --help` into Command Prompt.

![alt text](https://github.com/htleblond/ifcb_image_panner/blob/main/ifcb_image_panning_example_2.png)
