#coding=utf-8
#qpy:kivy

import os
pyhome=os.popen('echo $PYTHONHOME').read().strip()
opencv3=os.path.join(pyhome,'lib/python2.7/site-packages/cv2.py')
if not os.path.exists(opencv3):
    command="echo 'from cv.cv2 import *'>%s"%opencv3
    os.system(command)

import kivy
import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture

face_cascade=cv2.CascadeClassifier(r'./lbpcascade_frontalface.xml')
cv2.setUseOptimized(True)

#
class CV2mera(Camera):
    def _camera_loaded(self, *largs):
        if kivy.platform=='android':
            self.texture = Texture.create(size=self.resolution,colorfmt='rgb')
            self.texture_size = list(self.texture.size)
        else:
            super(CV2mera, self)._camera_loaded()

    def on_tex(self, *l):
        if kivy.platform=='android':
            buf = self._camera.grab_frame()
            if not buf:
                return
            frame = self._camera.decode_frame(buf)
            buf = self.process_frame(frame)
            self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        super(CV2mera, self).on_tex(*l)

    def process_frame(self,frame):
        r,g,b=cv2.split(frame)
        frame=cv2.merge((b,g,r))        
        rows,cols,channel=frame.shape
        M=cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        dst=cv2.warpAffine(frame,M,(cols,rows))
        frame=cv2.flip(dst,1)
        if self.index==1:
            frame=cv2.flip(dst,-1)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.20,
        minNeighbors = 1,
        minSize = (1,1),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+w),(0,255,0),2)
        
        return frame.tostring()

class MyLayout(BoxLayout):
    pass
class MainApp(App):
    def build(self):
        return MyLayout()
MainApp().run()