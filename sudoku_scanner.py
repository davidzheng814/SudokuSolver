


'''
    TODO: Look at other options for selecting number with K-nearest neighbors
    Such as, don't use same number twice in row column or grid. 

'''







import numpy as np
import cv2, cv
from matplotlib import pyplot as plt
import math
import sudoku
import sys

def show_image(img,save_as=None):
    '''
    displays image
    img - image to be displayed
    save_as - name of file if image should be saved
    '''
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',img)
    k=cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite(save_as,img)
        cv2.destroyAllWindows()


def floodfill(img,y_0,x_0,newval,min_bound=0,max_bound=255,dim=[-1,-1,-1,-1]):
    '''
    floodfills a grayscale image with a given pixel value
    img - image to be floodfilled
    y_0, x_0 - starting point
    newval - new pixel value
    (min_bound, max_bound) - range of pixel values that floodfill can fill with new value
    '''

    height = img.shape[0]
    width = img.shape[1]
    stack=[(y_0,x_0)]
    while len(stack)>0:
        y,x = stack.pop()

        if y<0 or y>=height or x<0 or x>=width:
            continue
        else:
            if img[y][x]>=min_bound and img[y][x]<=max_bound:
                img[y][x]=newval
                if y<dim[0]:
                    dim[0]=y
                if y>dim[2]:
                    dim[2]=y
                if x<dim[1]:
                    dim[1]=x
                if x>dim[3]:
                    dim[3]=x
                stack.append((y-1,x))
                stack.append((y,x-1))
                stack.append((y+1,x))
                stack.append((y,x+1))
def drawlines(img,lines):
    '''
    draw lines on image
    img - the image lines are drawn on
    lines - list of rho, theta tuples written in standard form
    '''
    height = img.shape[0]
    width = img.shape[1]
    for i,(rho,theta) in enumerate(lines): #find the two endpoints of each line
        point1=[0,0]
        point2=[0,0]
        if theta>cv.CV_PI*45/180.0 and theta<cv.CV_PI*135/180: #theta is ~90 degrees, find points on horizontal line
            point1[1] = rho/math.sin(theta)
            point1[0] = 0
            point2[1] = -1*width/math.tan(theta) + rho/math.sin(theta)
            point2[0] = width
        else: #theta is ~0 degrees or ~180 degrees, find points on vertical line
            point1[0] = rho/math.cos(theta)
            point1[1] = 0
            point2[0] = -1*height*math.tan(theta) + rho/math.cos(theta)
            point2[1] = height
        color = 150
        cv2.line(img,(int(point1[0]),int(point1[1])),(int(point2[0]),int(point2[1])),color)
def generate_digits_training_set():
    '''
    parses 'digits.png' and generates an npz file with cv2 readable training data
    '''

    img = cv2.imread('digits.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
    # Make it into a Numpy array. 
    arr = np.array(cells)
    # Now we prepare train_data and test_data.
    train = arr.reshape(-1,400).astype(np.float32)
    # Create labels for train and test data
    k = np.arange(10)
    train_labels = np.repeat(k,500)[:,np.newaxis]
    np.savez('knn_data.npz',train=train, train_labels=train_labels)

def add_answer_files(start, end):
    while start<= end:
        print start
        f = open('training_sets/train'+str(start)+".txt",'w')
        for i in range(9):
            inp = raw_input("")
            f.write(str(inp) + "\n")

        f.close()
        start += 1
def generate_custom_training_set(img_name, answer_file, sd):
    '''
    use Sudoku puzzle to create a numpy training set 
    img_name - must be a valid img file name
    answer_file - solution of the puzzle, in small board format
    '''
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    puzzle_size = 9
    img = cv2.resize(img, (puzzle_size*24,puzzle_size*24))

    if sd != 0:
        kernel=0
        img = cv2.GaussianBlur(img,(sd,sd),kernel)

        blur_window = 5
        subtract_from_mean = 2
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blur_window,
            subtract_from_mean)

    cells = [np.hsplit(row,puzzle_size) for row in np.vsplit(img,puzzle_size)]
    for i in range(puzzle_size):
        for j in range(puzzle_size):
            cells[i][j] = cells[i][j][2:-2, 2:-2]
    arr = np.array(cells)
    train = arr.reshape(-1,400).astype(np.float32)

    train_labels = []
    f=open(answer_file,'r')
    for i in range(puzzle_size):
        line = f.readline()
        for j in range(puzzle_size):
            char = line[j]
            if char == 'o':
                train_labels.append([0])
            else:
                train_labels.append([int(char)])

    train_labels = np.array(train_labels)
    np.savez(img_name[:-3]+'npz', train = train, train_labels = train_labels)
def train_knn(file_names, knn = None):
    '''
    returns knn that is trained with handwritten images
    file_names - must be a list of npz files
    '''
    file_name = file_names[-1]
    with np.load(file_name) as data:
        
        train = data['train']
        train_labels = data['train_labels']
        # Initiate kNN, train the data, then test it with test data for k=1
        if knn is None:

            knn = cv2.KNearest()
            knn.train(train,train_labels, updateBase = False)
        else:
            knn.train(train,train_labels, updateBase = True)

        if len(file_names) == 1:
            return knn
        else:
            file_names.pop()
            return train_knn(file_names,knn)
    raise

def recognize_digit(knn, img):
    '''
    converts img to list of digits
    knn - trained KNearestNeighbor object
    img - image to be converted
    '''
    puzzle_size = 9
    img = cv2.resize(img, (puzzle_size*24,puzzle_size*24))
    cells = [np.hsplit(row,puzzle_size) for row in np.vsplit(img,puzzle_size)]
    for i in range(puzzle_size):
        for j in range(puzzle_size):
            cells[i][j] = cells[i][j][2:-2, 2:-2]

    arr = np.array(cells)
    arr = arr.reshape(-1,400).astype(np.float32)
    ret,result,neighbors,dist = knn.find_nearest(arr,k=3)
    return result
def scan(img_name):

    '''
    generates a Sudoku puzzle from a given image
    img_name - name of image of Sudoku puzzle
    returns Puzzle object
    '''
    img = cv2.imread(img_name,0)
    refactor = 9*24*2/float(img.shape[1])
    img = cv2.resize(img,dsize= (0,0), fx= refactor,fy= refactor)
    original_img = img
    height = img.shape[0]
    width = img.shape[1]

    #blur image
    kernel=0
    sd=11
    img = cv2.GaussianBlur(img,(sd,sd),kernel)
    #apply threshold
    blur_window = 5
    subtract_from_mean = 2
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blur_window,
        subtract_from_mean)

    #store the black-white image for later
    black_white_img = img
    #invert, edges become white
    img = 255-img
    show_image(img)
    #dilate edges
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    #img = cv2.dilate(img,kernel)

    #find frame
    max_area = 0
    max_point = (0,0)
    temp_shade = 60
    for y in range(height): #find largest blob through floodfill
        for x in range(width):
            if img[y][x]>=temp_shade+1:
                dim=[height,width,-1,-1]
                floodfill(img,y,x,temp_shade,min_bound=temp_shade+1,dim=dim)

                area=(dim[3]-dim[1])*(dim[2]-dim[0])
                if area>max_area:
                    max_area = area
                    max_point = (y,x) 
    floodfill(img,max_point[0],max_point[1],255,min_bound=temp_shade-1,max_bound=temp_shade+1) #color largest blob white
    
    for y in range(height): #color other blobs black
        for x in range(width):
            if img[y][x]==temp_shade:
                img[y][x]=0


    #erode frame
    # cv2.erode(black_white_img,kernel,iterations=1)

    #find hough lines
    rho_accuracy = 1
    theta_accuracy = cv.CV_PI/180.0
    threshold = 200

    lines = cv2.HoughLines(img,rho_accuracy, theta_accuracy, threshold)[0]
    for i,(rho,theta) in enumerate(lines): #adjust all negative rhos to positive rhos
        if rho<0:
            lines[i][0] = -rho
            lines[i][1] = theta-cv.CV_PI
    drawlines(img, lines)
    show_image(img)
    #merge lines
    for i,(rho,theta) in enumerate(lines):
        if theta == -100:
            continue
        for j,(rho2, theta2) in enumerate(lines[i+1:],start=i+1):
            if theta2 == -100:
                continue
            if math.fabs(rho-rho2)<20 and math.fabs(theta-theta2)<cv.CV_PI*10/180.0:
                #if (points[i][0][0] - points[j][0][0])**2 + (points[i][0][1] - points[j][0][1])**2 <64*64 and (points[i][1][0] - points[j][1][0])**2 + (points[i][1][1] - points[j][1][1])**2 <64*64:
                lines[i][0] = (rho+rho2)/2
                lines[i][1] = (theta+theta2)/2
                lines[j][0] = 0
                lines[j][1] = -100

    #find boundary lines
    top_edge=None
    bottom_edge=None
    left_edge=None
    left_edge_xintercept = None
    right_edge=None
    right_edge_xintercept = None
    for i,(rho,theta) in enumerate(lines):
        if theta == -100:
            continue
        
        if theta>cv.CV_PI*80/180 and theta<cv.CV_PI*100/180: #line is nearly vertical
            if top_edge is None or rho < top_edge[0]:
                top_edge = [rho,theta]
            if bottom_edge is None or rho > bottom_edge[0]:
                bottom_edge = [rho, theta]
        elif theta<cv.CV_PI*10/180 or theta>cv.CV_PI*170/180: #line is nearly horizontal
            x_intercept = rho/math.cos(theta)
            if left_edge is None or x_intercept<left_edge_xintercept:
                left_edge_xintercept = x_intercept
                left_edge = [rho, theta]
            if right_edge is None or x_intercept>right_edge_xintercept:
                right_edge_xintercept = x_intercept
                right_edge = [rho, theta]

    #check if correct number of edges found
    if top_edge is None or bottom_edge is None or left_edge is None or right_edge is None: 
        return False


    #find intersection points of edges:
    intersect = []
    for edge1, edge2 in [(top_edge,left_edge),(top_edge,right_edge),(bottom_edge,left_edge),(bottom_edge,right_edge)]:
        rho1 = edge1[0]
        theta1 = edge1[1]
        rho2 = edge2[0]
        theta2 = edge2[1]
        if theta2==0:
            x = rho2
            x = min(max(x,0),width-1)
            y = -x/math.tan(theta1) + rho1*math.cos(theta1)/math.tan(theta1) + rho1*math.sin(theta1) 
            y = min(max(y,0),height-1)
            intersect.append((y,x))
        else:
            x = (rho2*math.cos(theta2)/math.tan(theta2)+rho2*math.sin(theta2) - rho1*math.cos(theta1)/math.tan(theta1) - rho1*math.sin(theta1))/(1/math.tan(theta2) - 1/math.tan(theta1))
            x = min(max(x,0),width-1)
            y = -x/math.tan(theta1) + rho1*math.cos(theta1)/math.tan(theta1) + rho1*math.sin(theta1)
            y = min(max(y,0),height-1)
            intersect.append((y,x))

    #find length of longest edge
    distance = lambda pt1,pt2: (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2
    max_length_sq = max(distance(intersect[0],intersect[1]),distance(intersect[0],intersect[2]),distance(intersect[1],intersect[3]),distance(intersect[2],intersect[3]))
    max_length = math.sqrt(max_length_sq)

    #adjust image to be of size max_length*max_length and top-left corner to be at intersect[0]
    img = black_white_img[intersect[0][0]:intersect[0][0]+max_length,intersect[0][1]:intersect[0][1]+max_length]
    show_image(img)
    height = img.shape[0]
    width = img.shape[1]

    #initiate OCR training data
    training_sets = []
    for i in range(1,16):
        training_sets.append('training_sets/train'+str(i)+'.npz')
    knn = train_knn(training_sets)

    #detect digits of puzzle
    digits = recognize_digit(knn, img)

    #convert to a replace list for Sudoku puzzle
    puzzle_size = 9
    count = 0
    replace_list = []
    for i in range(puzzle_size):
        for j in range(puzzle_size):
            digit = int(digits[count])
            if 0 < digit <= puzzle_size:
                replace_list.append((i,j,digit))
            count+=1

    #print board
    count = 0
    text = ""
    for i in range(puzzle_size):
        for j in range(puzzle_size):
            digit = int(digits[count])
            if digit == 0:
                text += "o"
            else:
                text += str(digit)
            count +=1
        text += "\n"

    print text


    return sudoku.Puzzle(inp = replace_list)

for blur_sd in [11,9,13,0,7,15]:
    try:
        for i in range(1,16):
            generate_custom_training_set('training_sets/train'+str(i)+'.png','training_sets/train'+str(i)+'.txt', blur_sd)
        puzzle = scan(sys.argv[1])
        puzzle.solveBoardWithGuessing()
        puzzle.printSmallBoard()
        break
    except ValueError:
        print "Failure with blur: " + str(blur_sd)
    except IndexError:
        print "Failure with blur: " + str(blur_sd)
