import numpy as np

def calculo_mar(face,p_boca):

    try:

         face = np.array([[coord.x, coord.y] for coord in face])
         
         face_boca = face[p_boca, :]

         mar = (np.linalg.norm(face_boca[0] - face_boca[1])
                + np.linalg.norm(face_boca[2] - face_boca[3])
                + np.linalg.norm(face_boca[4] - face_boca[5])) / (2*(np.linalg.norm(face_boca[6] - face_boca[7])))
   
    except:
         
         mar = 0.0
   
    return mar