# SocialDistanceSSDRaspberry
<br />
Find out more about this work at : https://link.springer.com/chapter/10.1007%2F978-3-030-66840-2_91
<br />
<br />
Description :: This application is about a Social Distance Monitoring system using Raspberry Pi based on SSD pretrained model.
<br /><br />

<br />

## Requirements
On a fresh installation of the Raspberry Pi 3 or 4 install:<br />

        sudo pip3 install imutils
        sudo pip3 install opencv-python opencv-contrib-python
        sudo apt-get install libhdf5-103
        sudo apt-get install libatlas-base-dev
        sudo apt-get install libjasper-dev
        sudo apt-get update
        sudo apt-get install libqtgui4
        sudo apt-get install libqt4-test
        sudo pip3 install scipy
<br />

## Launch
Start social distance detection script : <br />

        # LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1.2.0 python3 real_time_ssd_detector.py
