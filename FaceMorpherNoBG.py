import numpy as np
import cv2
import dlib
print("DLIB CUDA:"+ str(dlib.DLIB_USE_CUDA))
class NoFaceFound(Exception):
   """Raised when there is no face found"""
   pass

def calculate_margin_help(img1,img2):
    size1 = img1.shape
    size2 = img2.shape
    diff0 = abs(size1[0]-size2[0])//2
    diff1 = abs(size1[1]-size2[1])//2
    avg0 = (size1[0]+size2[0])//2
    avg1 = (size1[1]+size2[1])//2

    return [size1,size2,diff0,diff1,avg0,avg1]

def crop_image(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)

    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        scale0 = size1[0]/size2[0]
        scale1 = size1[1]/size2[1]
        if(scale0 > scale1):
            res = cv2.resize(img2,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img2,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(img1,res)

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        scale0 = size2[0]/size1[0]
        scale1 = size2[1]/size1[1]
        if(scale0 > scale1):
            res = cv2.resize(img1,None,fx=scale0,fy=scale0,interpolation=cv2.INTER_AREA)
        else:
            res = cv2.resize(img1,None,fx=scale1,fy=scale1,interpolation=cv2.INTER_AREA)
        return crop_image_help(res,img2)

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        return [img1[diff0:avg0,:],img2[:,-diff1:avg1]]
    
    else:
        return [img1[:,diff1:avg1],img2[-diff0:avg0,:]]

def crop_image_help(img1,img2):
    [size1,size2,diff0,diff1,avg0,avg1] = calculate_margin_help(img1,img2)
    
    if(size1[0] == size2[0] and size1[1] == size2[1]):
        return [img1,img2]

    elif(size1[0] <= size2[0] and size1[1] <= size2[1]):
        return [img1,img2[-diff0:avg0,-diff1:avg1]]

    elif(size1[0] >= size2[0] and size1[1] >= size2[1]):
        return [img1[diff0:avg0,diff1:avg1],img2]

    elif(size1[0] >= size2[0] and size1[1] <= size2[1]):
        return [img1[diff0:avg0,:],img2[:,-diff1:avg1]]

    else:
        return [img1[:,diff1:avg1],img2[diff0:avg0,:]]

def generate_face_correspondences(theImage1, theImage2, detector,predictor):
    # Detect the points of face.
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    corresp = np.zeros((68,2))

    imgList = crop_image(theImage1,theImage2)
    list1 = []
    list2 = []
    j = 1

    for img in imgList:

        size = (img.shape[0],img.shape[1])
        if(j == 1):
            currList = list1
        else:
            currList = list2

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        # start = time.time()
        dets = detector(img, 0)
        # end = time.time()
        # print("dets: "+str(end - start)+"s")
        try:
            if len(dets) == 0:
                raise NoFaceFound
        except NoFaceFound:
            print("Sorry, but I couldn't find a face in the image.")

        j=j+1
        for k, rect in enumerate(dets):
            # Get the landmarks/parts for the face in rect.
            shape = predictor(img, rect)
            # corresp = face_utils.shape_to_np(shape)
            
            for i in range(0,68):
                x = shape.part(i).x
                y = shape.part(i).y
                currList.append((x, y))
                corresp[i][0] += x
                corresp[i][1] += y



    # Add back the background
    narray = corresp/2

    
    return [size,imgList[0],imgList[1],list1,list2,narray]

# Check if a point is inside a rectangle
def rect_contains(rect, point):

    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Write the delaunay triangles into a file
def draw_delaunay(f_w, f_h, subdiv, dictionary1):

    list4 = []

    triangleList = subdiv.getTriangleList()
    r = (0, 0, f_w, f_h)

    for t in triangleList :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            list4.append((dictionary1[pt1],dictionary1[pt2],dictionary1[pt3]))

    dictionary1 = {}
    return list4

def make_delaunay(f_w, f_h, theList, img1, img2):

    # Make a rectangle.
    rect = (0, 0, f_w, f_h)

    # Create an instance of Subdiv2D.
    subdiv = cv2.Subdiv2D(rect)

    # Make a points list and a searchable dictionary. 
    theList = theList.tolist()
    points = [(int(x[0]),int(x[1])) for x in theList]
    dictionary = {x[0]:x[1] for x in list(zip(points, range(76)))}
    
    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)

    # Make a delaunay triangulation list.
    list4 = draw_delaunay(f_w, f_h, subdiv, dictionary)
   
    # Return the list.
    return list4

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def apply_affine_transform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morph_triangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = apply_affine_transform(img1Rect, t1Rect, tRect, size)
    warpImage2 = apply_affine_transform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask
    
    return img


def generate_morph_sequence(img1,img2,points1,points2,tri_list,size, alpha):


    # Convert Mat to float data type
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # Read array of corresponding points
    points = []

    # Compute weighted average point coordinates
    for i in range(0, len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((x,y))
    
    # Allocate space for final output
    morphed_frame = np.zeros(img1.shape, dtype = img1.dtype)




    for i in range(len(tri_list)):    
        x = int(tri_list[i][0])
        y = int(tri_list[i][1])
        z = int(tri_list[i][2])
        
        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x], points[y], points[z]]

        # Morph one triangle at a time.
        morphed_frame = morph_triangle(img1, img2, morphed_frame, t1, t2, t, alpha)

    
        
    return morphed_frame

def doMorphing(img1, img2):
    [size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2, predictorPath)
    # tri = make_delaunay(size[1], size[0], list3, img1, img2)
    # morphed_frame = generate_morph_sequence(img1, img2, points1, points2, tri, size, alpha)
    # return morphed_frame
    return img1, img2



def merge2images(img1,img2,det,pred,alpha=0.5):
    predictorPath ='shape_predictor_68_face_landmarks.dat'
    filename1 = img1
    filename2 = img2


    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    # start = time.time()
    [size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2, det,pred)

    # end = time.time()
    # print("CORR: "+str(end - start)+"s")

    
    # start = time.time()
    tri = make_delaunay(size[1], size[0], list3, img1, img2)


    # end = time.time()
    # print("delaunay: "+str(end - start)+"s")

    # start = time.time()
    output = generate_morph_sequence(img1, img2, points1, points2, tri, size, alpha)

    # end = time.time()
    # print("morph: "+str(end - start)+"s")
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return np.uint8(output)









