# -*- coding: utf-8 -*-
from numpy.lib.utils import source
import detect
import uuid
import io, tempfile, os, shutil
from PIL import Image, ImageFilter

class Main(object):
    def __init__(self):
        #print("working directory" + os.getcwd())
        #for f in os.listdir("/bin/"):
        #    print( "/bin/" +f)
        self.rootdir = os.getcwd()
        if not os.path.isdir(self.rootdir+"/runs/detect"):
            os.mkdir(self.rootdir +"/runs")
            os.mkdir(self.rootdir +"/runs/detect")

    def predict(self, skill_input):
        bytes_buf = io.BytesIO(skill_input)
        uname = str(uuid.uuid4())
        tempfile = uname + ".png"
        with open(tempfile, "wb") as tempf:
            tempf.write( bytes_buf.read())
        
        im = Image.open( tempfile)
        imgsize = [ im.height, im.width]
        
        detect.run( weights='models/samsung_logo_1121.pt', name=uname, save_conf=True, 
                   source=tempfile, imgsz=imgsize, device='cpu', conf_thres=0.5, 
                   project=self.rootdir+'/runs/detect', save_txt=True )
        os.remove(uname+".png")
        result = { }
        result['logos'] = []
        print(os.getcwd())
        # no logo detected... 
        if not os.path.exists(self.rootdir+"/runs/detect/"+uname+"/labels/" + uname + ".txt"):
            result["message"] = "nothing is detected"
            shutil.rmtree(self.rootdir+"/runs/detect/"+uname, ignore_errors=True)
            return result
        with open(self.rootdir+"/runs/detect/"+uname+"/labels/" + uname + ".txt", "r") as outf:
            for l in outf.readlines():
                obj = {}
                items = l.split()
                obj["class"] = items[0]
                obj["confidence"] = float(items[5])
                obj["x"] = int(im.width * float(items[1]))
                obj["y"] = int(im.height * float(items[2]))
                obj["width"] = int(im.width * float(items[3]))
                obj["height"] = int(im.height * float(items[4]))
                result["logos"].append( obj)
        result["message"] = "ok"
        print( result)
        shutil.rmtree(self.rootdir+"/runs/detect/"+uname, ignore_errors=True)
        return result



if __name__ == '__main__':
	m = Main()
	with open("./samples/Test.png", "rb") as input_f:
		print( m.predict( input_f.read()))
