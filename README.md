# about Project

1. Generated Synthetic data using isaac-sim
- make scene
	- using Replicator

2. Data Preprocessing for training

3. Train Both Model, Yolo and UNet
- YOLO : object detection in image data
- UNet : make grid map using image data (Map scale can be changed)

4. Modify ultralytics cpp source code to using our both model
- In order to, Increase Realtime Inference Performance for Jetson Nano
- Compile src code

5. Run script
