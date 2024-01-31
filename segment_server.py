"""
    TCP server to inference SAM remotely
"""

print("Starting SAM server..." );

import sys
import time
import json
import math
import itertools
import socket
import numpy as np
import cv2

HOST=""
PORT=8727
#PORT = sys.argv[1];

#model path
sam_checkpoint = "../models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

#HELPER
class TicToc:
    t0 = 0;
    #start timer
    def tic(self, msg = None):
        self.t0 = time.monotonic();
        if(msg is not None):
            print(msg + "...", end="")

    #stop timer
    def toc(self):
        dt = time.monotonic() - self.t0;
        print("("+str(round(dt,2))+" s)");
        

tt = TicToc();

#LOADING LIBRARIES
tt.tic("Loading libraries");
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
tt.toc();

#LOAD MODELS
tt.tic("Loading SAM");
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
tt.toc();

#SERVER
print(f"Starting server...")
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
    s.bind((HOST,PORT))
    while True:
        try:
            print(f"Listerning on port {PORT}...")
            s.listen()
            con,client_addr = s.accept()
            with con:
                print(f"Connected by {client_addr}")
                connected=True
                while connected:
                    try:
                        jstr="";
                        #get a line
                        while connected:
                            data = con.recv(1)
                            c = chr(data[0])
                            #found new line, end of cmd
                            if(c=='\r' and chr(con.recv(1)[0])=='\n'):
                                break;
                            jstr+=c

                        if jstr:
                            print(">"+jstr)
                            cmd = json.loads(jstr)
                            if(cmd["name"]=="predict"):
                                imgsz = cmd["size"]
                                buf = bytearray(imgsz)
                                boxes =  np.array(cmd["box"]) if "box" in cmd else None
                                points = np.array(cmd["point"]) if "point" in cmd else None
                                labels = np.array(cmd["label"]) if "label" in cmd else None
                                multimask = cmd["multimask"] if "multimask" in cmd else False
                                con.sendall("OK\n".encode())
                                byte_reads = con.recv_into(buf,imgsz)
                                if(byte_reads == imgsz):
                                    #decode image
                                    img = cv2.imdecode(np.frombuffer(buf,dtype=np.uint8), cv2.IMREAD_COLOR)
                                    #set image
                                    tt.tic("Inferencing");
                                    predictor.set_image(img);
                                    tt.toc();

                                    masks, scores, logits = predictor.predict(
                                        point_coords=points,
                                        point_labels=labels,
                                        box = boxes,
                                        multimask_output=multimask,
                                    )

                                    id = scores.tolist().index(max(scores))
                                    mask = masks[id].astype(np.uint8)*255;
                                    maskbuf = cv2.imencode(".png",mask)[1].tobytes()
                                    con.sendall(f'{{"mask":{id},"score":{scores[id]},"size":{len(maskbuf)}}}\n'.encode())
                                    con.sendall(maskbuf)
                                    #img[mask] = (0,255,0)
                                    #for p in points:
                                    #    cv2.circle(img,p,5,(255,0,255),-1)
                                    #cv2.imshow("img",img)
                                    #cv2.waitKey(1)
                                    con.sendall("OK\n".encode())
                                else:
                                   raise Exception(f"Corrupted data when processing {jstr}"); 
                            else:
                                print(f"Unknown command {jstr}")
                    except Exception as ex:
                        print("ERROR: " + str(ex))
                        con.sendall(("ERROR: "+str(ex)+"\n").encode())
                    time.sleep(0.010); #100 hz polling
        except Exception as ex: 
            print("Opps! "+str(ex))





