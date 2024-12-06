import cv2 as cv
import numpy as np
import math
import os
from shapely.geometry import Polygon
import xlwt
file = xlwt.Workbook('encoding = utf-8')
sheet1 = file.add_sheet('sheet1',cell_overwrite_ok=True)


def Convolutional(raw_img, kernel):
    img = np.zeros((raw_img.shape[0],raw_img.shape[1])).astype(np.uint8)
    for x in range(1, raw_img.shape[0] - 1):
        for y in range(1, raw_img.shape[1] - 1):
            a = np.zeros([3, 3]).astype(np.uint8)
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    a[i, j] = kernel[i, j] * raw_img[x - 1 + i, y - 1 + j]
            img[x, y] = np.sum(a)

    return img

def image_binarization(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    retval, dst = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
    return dst

def RHM(label, img_connect):
    X = 0
    Y = 0
    for i in range(1, label):
        x, y = np.where(img_connect == i)
        X = (np.max(x) + np.min(x)) / 2 + X
        Y = (np.max(y) + np.min(y)) / 2 + Y
    return X, Y

def agfun(img):
    global i1 ,i2 ,i3 ,i4
    maxx = img.shape[0] / 2
    minx = img.shape[0] / 2
    maxy = img.shape[1] / 2
    miny = img.shape[1] / 2

    for i in range(1, label):
        x, y = np.where(img_connect == i)
        if np.min(x) < minx:
            minx = np.min(x)
            i1 = i
        if np.min(y) < miny:
            miny = np.min(y)
            i2 = i
        if np.max(x) > maxx:
            maxx = np.min(x)
            i3 = i
        if np.max(y) > maxy:
            maxy = np.min(y)
            i4 = i

    x1, y1 = np.where(img_connect == i1)
    points=[]
    for i in range(x1.shape[0]):
        points.append((x1[i],y1[i]))
    polygon = Polygon(points)
    ConvexHull1 = polygon.convex_hull.area
    x2, y2 = np.where(img_connect == i2)
    points2 = []
    for i in range(x2.shape[0]):
        points2.append((x2[i], y2[i]))
    polygon = Polygon(points2)
    ConvexHull2 = polygon.convex_hull.area
    x3, y3 = np.where(img_connect == i3)
    points3 = []
    for i in range(x3.shape[0]):
        points3.append((x3[i], y3[i]))
    polygon = Polygon(points3)
    ConvexHull3 = polygon.convex_hull.area
    x4, y4 = np.where(img_connect == i4)
    points4 = []
    for i in range(x4.shape[0]):
        points4.append((x4[i], y4[i]))
    polygon = Polygon(points4)
    ConvexHull4 = polygon.convex_hull.area
    return ConvexHull1+ConvexHull2+ConvexHull3+ConvexHull4

def bubble(n, list):
    for i in range(0, len(list) - n - 1):
        temporary_variable = 0
        if list[i] > list[i + 1]:
            temporary_variable = list[i + 1]
            list[i + 1] = list[i]
            list[i] = temporary_variable
    n = n + 1
    if n != len(list) - 1:
        bubble(n, list)
    else:
        return list
def num(img):
    n=int(0)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y] != 0:
                n += 1
    return n



##
train_img = os.path.join("./img")
train_image_names = os.listdir(train_img)

for kk in range(0, len(train_image_names)):
    aaa = train_image_names[kk][0:-4]
    img = cv.imread(train_img + "/" + train_image_names[kk])
    img = image_binarization(img)
    gray_lap = cv.Laplacian(img, cv.CV_16S, ksize=3)
    img = cv.convertScaleAbs(gray_lap)
    cv.waitKey()
    label, img_connect = cv.connectedComponents(img)
    for i in range(1, label):
        if np.sum(img_connect == i) < 30:
            x, y = np.where(img_connect == i)
            for j in range(x.shape[0]):
                img[x[j], y[j]] = 0

    label, img_connect = cv.connectedComponents(img)
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]) / 9
    raw_img2 = Convolutional(img_connect, kernel)
    for x in range(img_connect.shape[0]):
        for y in range(img_connect.shape[1]):
            if raw_img2[x,y]==img_connect[x,y]:
                img_connect[x, y]=0

    sizex = int(img.shape[0])
    sizey = int(img.shape[1])
    img1 = img[0:int(sizex / 2), 0:int(sizey / 2)]
    img2 = img[0:int(sizex / 2), int(sizey / 2):int(sizey)]
    img4 = img[int(sizex / 2):int(sizex), int(sizey / 2):int(sizey)]
    img3 = img[int(sizex / 2):int(sizex), 0:int(sizey / 2)]
    label1, img_connect1 = cv.connectedComponents(img1)
    label2, img_connect2 = cv.connectedComponents(img2)
    label3, img_connect3 = cv.connectedComponents(img3)
    label4, img_connect4 = cv.connectedComponents(img4)

    # 1Balance

    imgT = img[int(0):int(sizex / 2), int(0):int(sizey)]
    imgB = img[int(sizex / 2):int(sizex), int(0):int(sizey)]
    imgR = img[int(0):int(sizex), int(sizey / 2):int(sizey)]
    imgL = img[int(0):int(sizex), int(0):int(sizey / 2)]
    labelT, img_connectT = cv.connectedComponents(imgT)
    labelB, img_connectB = cv.connectedComponents(imgB)
    labelR, img_connectR = cv.connectedComponents(imgR)
    labelL, img_connectL = cv.connectedComponents(imgL)
    wT = 0
    wB = 0
    wL = 0
    wR = 0
    wR2 = []
    wL2=[]
    p=0
    for i in range(1, labelT):
        x, y = np.where(img_connectT == i)
        wT = wT + np.sum(imgT.shape[0] - x)
    for i in range(1, labelB):
        x, y = np.where(img_connectB == i)
        wB = wB + np.sum(x)
    for i in range(1, labelL):
        x, y = np.where(img_connectL == i)
        wL = wL + np.sum(imgL.shape[1] - y)
    for i in range(1, labelR):
        x, y = np.where(img_connectR == i)
        wR = wR + np.sum(y)
    for i in range(-labelR,-1):
        x, y = np.where(img_connectR == -i)
        wR2.append(np.sum(y))
    for i in range(1, labelL):
        x, y = np.where(img_connectL == i)
        wL2.append(np.sum(imgL.shape[1] - y))

    D1X = (wT - wB) / (max(wT, wB))
    D1Y = (wL - wR) / (max(wL, wR))
    D1 = 1 - (abs(D1X) + abs(D1Y)) / 2
    print(f'1Balance:{D1}')

    # 2Symmetry
    wy2=0
    wx2=0
    ansewer=[]
    for i in range(1, min(labelT,labelB)):
        x1, y1 = np.where(img_connectT == i)
        x2, y2 = np.where(img_connectB == i)
        wy2 = wy2 + abs(np.sum(imgT.shape[0] - x1)-np.sum(imgB.shape[0] - x2))
    for i in range(1, min(labelL,labelR)):
        x1, y1 = np.where(img_connectL == i)
        x2, y2 = np.where(img_connectR == i)
        wx2 = wx2 + abs(np.sum(imgL.shape[1] - y1)-np.sum(imgR.shape[1] - y2))

    for i in range(0,min(len(wR2),len(wL2))):
        ansewer.append(abs(wL2[i]-wR2[i]))

    D2X = (wx2) / (max(wT, wB))
    D2Y = (wy2) / (max(wL, wR))
    D2R = (sum(ansewer)) / (max(wL, wR))
    D2 = abs(1 - (abs(D2X) + abs(D2Y) + abs(D2R)) / 3)
    print(f'2Symmetry:{D2}')

    # 3Proportionality

    D3 = 0
    aa=np.ones((1,5))
    for i in range(1, label):
        x, y = np.where(img_connect == i)
        b = np.max(y)
        h = np.max(x)
        pi = min(h / b, b / h)
        aa[0,0] = abs(1 - pi)
        aa[0,2] = abs(1/1.414 - pi)
        aa[0,3] = abs(1/1.618 - pi)
        aa[0,4] = abs(1/1.732 - pi)
        aa[0,1] = abs(1 / 2 - pi)
        a=np.min(aa)
        if abs(a) > 0.5:
            a = 0.5
        else:
            a = abs(a)
        D3 = D3 + (1 - (a) / 0.5)
    D3=D3 / (i + 1)

    print(f'3Proportionality:{D3}')

    # 4Rhythmicity
    X = 0
    Y = 0
    X1, Y1 = RHM(label1, img_connect1)
    X2, Y2 = RHM(label2, img_connect2)
    X3, Y3 = RHM(label3, img_connect3)
    X4, Y4 = RHM(label4, img_connect4)
    X1Transpose = (X1 - min(X1, X2, X3, X4)) / (max(X1, X2, X3, X4) - min(X1, X2, X3, X4))
    X2Transpose = (X2 - min(X1, X2, X3, X4)) / (max(X1, X2, X3, X4) - min(X1, X2, X3, X4))
    X3Transpose = (X3 - min(X1, X2, X3, X4)) / (max(X1, X2, X3, X4) - min(X1, X2, X3, X4))
    X4Transpose = (X4 - min(X1, X2, X3, X4)) / (max(X1, X2, X3, X4) - min(X1, X2, X3, X4))
    Y1Transpose = (Y1 - min(Y1, Y2, Y3, Y4)) / (max(Y1, Y2, Y3, Y4) - min(Y1, Y2, Y3, Y4))
    Y2Transpose = (Y2 - min(Y1, Y2, Y3, Y4)) / (max(Y1, Y2, Y3, Y4) - min(Y1, Y2, Y3, Y4))
    Y3Transpose = (Y3 - min(Y1, Y2, Y3, Y4)) / (max(Y1, Y2, Y3, Y4) - min(Y1, Y2, Y3, Y4))
    Y4Transpose = (Y4 - min(Y1, Y2, Y3, Y4)) / (max(Y1, Y2, Y3, Y4) - min(Y1, Y2, Y3, Y4))
    D4x = (abs(X1Transpose - X2Transpose) + abs(X1Transpose - X4Transpose) + abs(X1Transpose - X3Transpose) + abs(
        X2Transpose - X4Transpose) + abs(X2Transpose - X3Transpose) + abs(X4Transpose - X3Transpose)) / 6
    D4y = (abs(Y1Transpose - Y2Transpose) + abs(Y1Transpose - Y4Transpose) + abs(Y1Transpose - Y3Transpose) + abs(
        Y2Transpose - Y4Transpose) + abs(Y2Transpose - Y3Transpose) + abs(Y4Transpose - Y3Transpose)) / 6
    D4 = 1 - (abs(D4y) + abs(D4x)) / 2
    print(f'4Rhythmicity:{D4}')

    # 5Sequentiality
    w1 = label1 * 4
    w2 = label2 * 3
    w3 = label3 * 2
    w4 = label4 * 1
    list = np.array([w1, w2, w3, w4])
    bubble(0, list)

    v1 = int(np.where(list == w1)[0][0]) + 1
    v2 = int(np.where(list == w2)[0][0]) + 1
    v3 = int(np.where(list == w3)[0][0]) + 1
    v4 = int(np.where(list == w4)[0][0]) + 1
    D5 = 1 - (abs(4 - v1) + abs(3 - v2) + abs(2 - v3) + abs(1 - v4)) / 8
    print(f'5Sequentiality:{D5}')

    # 6Cohesiveness
    maxx = img.shape[0] / 2
    minx = img.shape[0] / 2
    maxy = img.shape[1] / 2
    miny = img.shape[1] / 2

    for i in range(1, label):
        x, y = np.where(img_connect == i)
        if np.min(x) < minx:
            minx = np.min(x)
            i1 = i
        if np.min(y) < miny:
            miny = np.min(y)
            i2 = i
        if np.max(x) > maxx:
            maxx = np.min(x)
            i3 = i
        if np.max(y) > maxy:
            maxy = np.min(y)
            i4 = i
    x1, y1 = np.where(img_connect == i1)
    points=[]
    for i in range(x1.shape[0]):
        points.append((x1[i],y1[i]))
    polygon = Polygon(points)
    ConvexHull1 = polygon.convex_hull.area
    x2, y2 = np.where(img_connect == i2)
    points = []
    for i in range(x2.shape[0]):
        points.append((x2[i], y2[i]))
    polygon = Polygon(points)
    ConvexHull2 = polygon.convex_hull.area
    x3, y3 = np.where(img_connect == i3)
    points3 = []
    for i in range(x3.shape[0]):
        points.append((x3[i], y3[i]))
    polygon = Polygon(points)
    ConvexHull3 = polygon.convex_hull.area
    x4, y4 = np.where(img_connect == i4)
    points = []
    for i in range(x4.shape[0]):
        points.append((x4[i], y4[i]))
    polygon = Polygon(points)
    ConvexHull4 = polygon.convex_hull.area

    ao = (ConvexHull1+ConvexHull2+ConvexHull3+ConvexHull4)
    ag = agfun(img)
    maxx = img.shape[0] / 2
    minx = img.shape[0] / 2
    maxy = img.shape[1] / 2
    miny = img.shape[1] / 2
    for i in range(1, label):
        if i != i1 and i != i2 and i != i3 and i != i4:
            x, y = np.where(img_connect == i)
            if np.min(x) < minx:
                minx = np.min(x)
            if np.min(y) < miny:
                miny = np.min(y)
            if np.max(x) > maxx:
                maxx = np.min(x)
            if np.max(y) > maxy:
                maxy = np.min(y)

    a=0
    for i in range(1, label):
        if i != i1 and i != i2 and i != i3 and i != i4:
            x, y = np.where(img_connect == i)
            points = []
            for i in range(x.shape[0]):
                points.append((x[i], y[i]))
            polygon = Polygon(points)
            a = polygon.convex_hull.area/x.shape[0] + a


    D6 = abs(1 - (ag - a) / (ao - a))
    print(f'6Cohesiveness:{D6}')

    # 7Regularity

    DD = 4 * label
    for i in range(1, label - 1):
        x, y = np.where(img_connect == i)
        left1 = np.min(x)
        right1 = np.max(x)
        top1 = np.min(y)
        bottom1 = np.max(y)
        for j in range(i + 1, label):
            x, y = np.where(img_connect == j)
            left2 = np.min(x)
            right2 = np.max(x)
            top2 = np.min(y)
            bottom2 = np.max(y)
            l = abs(left1 - left2)
            r = abs(right1 - right2)
            t = abs(top1 - top2)
            b = abs(bottom1 - bottom2)
            if 0 <= l <= 5:
                DD = DD - 1
                if 0 <= r <= 5:
                    DD = DD - 1
                    if 0 <= t <= 5:
                        DD = DD - 1
                        if 0 <= b <= 5:
                            DD = DD - 1
    D7S = 1 - DD / (4 * label)
    label, img_connect, stats, centroids = cv.connectedComponentsWithStats(img)
    dd = 2 * label
    n = centroids.shape[0]
    a = centroids
    X = a[:, 0]
    Y = a[:, 1]
    for i in range(0, n - 1):
        x1 = X[i]
        y1 = Y[i]
        for j in range(i + 1, n):
            x2 = X[j]
            y2 = Y[j]
            xx = abs(x1 - x2)
            yy = abs(y1 - y2)
            if 0 <= xx <= 5:
                dd = dd - 1
                if 0 <= yy <= 5:
                    dd = dd - 1
    D7C = 1 - dd / (2 * label)
    D7 = (D7S + D7C) / 2
    print(f'7Regularity:{D7}')

    # 9Common Directionality
    corners = cv.goodFeaturesToTrack(img, 150, 0.3, 50)
    corners = np.intp(corners)
    data = corners[:, 0]

    ng = 0
    for i in corners:
        x, y = i.ravel()
        cv.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv.putText(img, f'{ng}', (x + 3, y + 3), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 0, 255), 1, cv.LINE_AA)
        ng += 1

    def calculate_slope(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        if x2 - x1 == 0:
            return float('inf')
        slope = (y2 - y1) / (x2 - x1)
        return slope

    def is_parallel(line1, line2, threshold):
        slope1 = calculate_slope(line1[0], line1[1])
        slope2 = calculate_slope(line2[0], line2[1])
        if abs(slope1 - slope2) < threshold:
            return True
        return False

    def count_parallel_lines(data, threshold):
        mts = 0
        mtg = 0
        groups = []

        for i in range(len(data)):
            added_to_group = False
            for j in range(i + 1, len(data)):
                line1 = (data[i], data[j])
                is_parallel_to_group = False
                for group in groups:
                    if is_parallel(line1, group[0], threshold):
                        group.append(line1)
                        mts += 1
                        is_parallel_to_group = True
                        added_to_group = True
                        break
                if not is_parallel_to_group:
                    groups.append([line1])
                    added_to_group = True
            if not added_to_group:
                mts += 1

        i = 0
        while i < len(groups):
            group = groups[i]
            if len(group) > 1:
                mtg += 1
                i += 1
            else:
                mts -= 1
                groups.pop(i)

        return mts, mtg

    threshold = 1.0

    mts, mtg = count_parallel_lines(data, threshold)
    D9 = (2 * (mts - mtg)) / ((ng * (ng - 1)))

    # 10Continuity

    def getLinearEquation(p1x, p1y, p2x, p2y):
        sign = 1
        a = p2y - p1y
        if a < 0:
            sign = -1
            a = sign * a
        b = sign * (p1x - p2x)
        c = sign * (p1y * p2x - p1x * p2y)
        return [a, b, c]

    def get_distance_from_point_to_line(point, line_point1, line_point2):
        A = line_point2[1] - line_point1[1]
        B = line_point1[0] - line_point2[0]
        C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
            (line_point2[0] - line_point1[0]) * line_point1[1]
        distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A ** 2 + B ** 2))
        return distance

    corners = cv.goodFeaturesToTrack(img, 150, 0.3, 50)
    corners = np.intp(corners)
    data = corners[:, 0]

    ng = 0
    for i in corners:
        x, y = i.ravel()
        cv.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv.putText(img, f'{ng}', (x + 3, y + 3), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 0, 255), 1, cv.LINE_AA)
        ng += 1

    distance = get_distance_from_point_to_line(data[2, :], data[1, :], data[2, :])
    mls = 0
    n_num = []
    mlsi = np.ones(len(data))
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            num = 0
            for k in range(len(data)):
                distance = get_distance_from_point_to_line(data[k, :], data[i, :], data[j, :])
                if distance > 0 and distance < 20:
                    mlsi[i] = 0
                    num += 1
            if num >= 3:
                n = num + 2
                n_num.append(n)
                mls += n / (n - 1)

    mls = ng - sum(mlsi)
    D10 = abs((mls) / ng)

    # 8Attractiveness

    D8 = 1 - (1 / ng)

    print(f'8Attractiveness:{D8}')
    print(f'9Common Directionality:{D9}')
    print(f'10Continuity:{D10}')

    #11Simplification
    def calculate_distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_nsi_nsg(img_connect, thresh):
        groups = []
        for i in range(1, label):
            x, y = np.where(img_connect == i)
            polygon = Polygon(zip(x, y))
            added_to_group = False
            for group in groups:
                for existing_polygon in group:
                    if polygon.intersects(existing_polygon) or polygon.contains(
                            existing_polygon) or existing_polygon.contains(polygon):
                        group.append(polygon)
                        added_to_group = True
                        break
                    # Check distance condition
                    elif calculate_distance(polygon.exterior.coords[0][0], polygon.exterior.coords[0][1],
                                            existing_polygon.exterior.coords[0][0],
                                            existing_polygon.exterior.coords[0][1]) < thresh:
                        group.append(polygon)
                        added_to_group = True
                        break
                if added_to_group:
                    break
            if not added_to_group:
                groups.append([polygon])

        nsi = 0
        nsg = len(groups)
        for group in groups:
            nsi += len(group)

        return nsi, nsg

    thresh = 30

    nsi_total = 0
    nsg_total = 0

    nsi, nsg = calculate_nsi_nsg(img_connect, thresh)
    nsi_total += nsi
    nsg_total += nsg

    D11 = (nsi - nsg) / label
    print(f'11Simplification:{D11}')

    # 12Similarity
    m = 20
    angle_threshold = 5

    img_connect_vectors = np.zeros((label, m))


    def calculate_similarity(img_connect_vectors, m):
        n = img_connect_vectors.shape[0]
        similarity_sum = 0

        for i in range(n):
            for j in range(i + 1, n):
                A_dot_B = np.dot(img_connect_vectors[i], img_connect_vectors[j])
                A_norm = np.linalg.norm(img_connect_vectors[i])
                B_norm = np.linalg.norm(img_connect_vectors[j])

                if A_norm == 0 or B_norm == 0:
                    continue

                Rij = A_norm / B_norm
                r = abs(Rij - 1)
                cos_theta = A_dot_B / (A_norm * B_norm)
                Dij = cos_theta / (r + 1)
                similarity_sum += Dij

        similarity = similarity_sum / (n * (n - 1) / 2)
        return similarity


    def calculate_width(x, y):
        min_x = np.min(x)
        max_x = np.max(x)
        min_y = np.min(y)
        max_y = np.max(y)
        width = max(max_x - min_x, max_y - min_y)
        return width


    widths = []
    for i in range(1, label):
        x, y = np.where(img_connect == i)
        width = calculate_width(x, y)
        widths.append(width)

    max_width = max(widths)
    scale_factor = m / max_width

    for i in range(1, label):
        x, y = np.where(img_connect == i)
        x_scaled = (x - np.mean(x)) * scale_factor + np.mean(x)
        y_scaled = (y - np.mean(y)) * scale_factor + np.mean(y)
        center_x = (np.max(x_scaled) + np.min(x_scaled)) / 2
        center_y = (np.max(y_scaled) + np.min(y_scaled)) / 2
        width = calculate_width(x_scaled, y_scaled)

        angles = np.linspace(0, 2 * math.pi, angle_threshold, endpoint=False)
        vectors = []
        for angle in angles:
            end_x = center_x + math.cos(angle) * width
            end_y = center_y + math.sin(angle) * width

            if int(end_x) < img_connect.shape[0] and int(end_y) < img_connect.shape[1]:
                vector = img_connect[int(end_x)][int(end_y)] - img_connect[i][j]
                vectors.append(vector)

        img_connect_vectors[i, :len(vectors)] = np.array(vectors)

    similarity = calculate_similarity(img_connect_vectors, m)
    D12 = similarity

    print(f'12Similarity:{D12}')

    # 13Proportional Similarity
    D13 = 0
    t = 0
    n = label
    for i in range(1, n - 1):
        x, y = np.where(img_connect == i)
        bi = np.max(y)
        hi = np.max(x)
        for j in range(i + 1, n):
            x, y = np.where(img_connect == j)
            bj = np.max(y)
            hj = np.max(x)
            tij = min((hi / bi) / (hj / bj), (hj / bj) / (hi / bi))
            t = t + tij
    D13 = 2 * t / ((n - 1) * n)
    print(f'13Proportional Similarity:{D13}')

    # 14Stability

    st_x, st_y, width, height = cv.boundingRect(img)
    bc = width
    hc = height

    M = cv.moments(img)
    XC = int(M["m10"] / M["m00"])
    YC = int(M["m01"] / M["m00"])
    xc = XC - (bc / 2)
    yc = (hc / 2) - YC

    n = label - 1
    y = 0
    for i in range(1, n - 1):
        a, b = np.where(img_connect == i)
        y1 = np.max(b)
        for j in range(i + 1, n):
            c, d = np.where(img_connect == j)
            y2 = np.max(d)
            if y1 > y2:
                y = y1
            else:
                y = y2
    Y = y - 100
    img1 = img[Y:y, :]
    st_x, st_y, width, height = cv.boundingRect(img1)
    xb = width
    D14y = 0.5 - (yc / hc)
    D14x = 1 - (2 * abs(xc) / bc)
    D14b = xb / bc
    D14 = (D14y + D14x + D14b) / 3
    print(f'14Stability:{D14}')

    # 15Hierarchy

    lengths = [stats[i, 4] for i in range(1, label)]
    sorted_lengths_idx = np.argsort(lengths)[::-1]

    S1 = lengths[sorted_lengths_idx[0]]
    threshold = 0.10 * S1

    hierarchy = []
    current_hierarchy = []

    for i in range(len(sorted_lengths_idx) - 1):
        current_hierarchy.append(sorted_lengths_idx[i])
        if abs(lengths[sorted_lengths_idx[i + 1]] - lengths[sorted_lengths_idx[i]]) > threshold:
            hierarchy.append(current_hierarchy)
            current_hierarchy = []

    hierarchy.append(current_hierarchy)

    d = 0
    for current_hierarchy in hierarchy:
        lengths_current_hierarchy = [lengths[i] for i in current_hierarchy]

        SiF = max(lengths_current_hierarchy) if lengths_current_hierarchy else 0

        SiL = min(lengths_current_hierarchy) if lengths_current_hierarchy else 0

        d += (SiF - SiL) / S1

    m = len(hierarchy)
    D15 = 1 - 1 / (2 * m) - d / 2

    print(f'15Hierarchy:{D15}')

    # 16Contrast

    average_lengths = [np.mean([lengths[i] for i in hierarchy[j]]) for j in range(len(hierarchy))]

    ratios = [average_lengths[i] / average_lengths[i - 1] for i in range(1, len(average_lengths))]

    average_ratio = np.mean(ratios)

    D16 = 1 - average_ratio

    print(f'16Contrast:{D16}')

    # 17Complexity

    length_ratios = []

    for i in range(1, label):
        points = np.argwhere(img_connect == i)

        hull = cv.convexHull(points)

        SiL = cv.arcLength(points, True)
        SiT = cv.arcLength(hull, True)

        SS = SiT / SiL if SiL > 0 else 0

        length_ratios.append(1 - SS)

    D17 = np.mean(length_ratios)

    print(f'17Complexity:{D17}')

    img = cv.imread(train_img + "/" + train_image_names[kk])
    cv.imwrite(f'./Result/{aaa}_D1-{round(D1,2)},D2-{round(D2,2)},D3-{round(D3,2)},D4-{round(D4,2)},,D5-{round(D5,2)},D6-{round(D6,2)},D7-{round(D7, 2)},D8-{round(D8, 2)},D9-{round(D9, 2)},D10-{round(D10, 2)},D11-{round(D11, 2)},D12-{round(D12, 2)},D13-{round(D13, 2)},D14-{round(D14, 2)},D15-{round(D15, 2)},D16-{round(D16, 2)},D17-{round(D17, 2)}.jpg', img,
               [int(cv.IMWRITE_JPEG_QUALITY), 100])
    img = image_binarization(img)
    img = cv.Canny(img, 150, 255)
    label, img_connect = cv.connectedComponents(img)
    for i in range(1, label):
        if np.sum(img_connect == i) < 30:  # 30
            x, y = np.where(img_connect == i)
            for j in range(x.shape[0]):
                img[x[j], y[j]] = 0
    cv.imwrite(
        f'./Result2/{aaa}_D1-{round(D1, 2)},D2-{round(D2, 2)},D3-{round(D3, 2)},D4-{round(D4, 2)},,D5-{round(D5, 2)},D6-{round(D6, 2)},D7-{round(D7, 2)},D8-{round(D8, 2)},D9-{round(D9, 2)},D10-{round(D10, 2)},D11-{round(D11, 2)},D12-{round(D12, 2)},D13-{round(D13, 2)},D14-{round(D14, 2)},D15-{round(D15, 2)},D16-{round(D16, 2)},D17-{round(D17, 2)}.jpg',
        img,
        [int(cv.IMWRITE_JPEG_QUALITY), 100])
    cv.waitKey()

    sheet1.write(kk, 0, round(D1, 3))
    sheet1.write(kk, 1, round(D2, 3))
    sheet1.write(kk, 2, round(D3, 3))
    sheet1.write(kk, 3, round(D4, 3))
    sheet1.write(kk, 4, round(D5, 3))
    sheet1.write(kk, 5, round(D6, 3))
    sheet1.write(kk, 6, round(D7, 3))
    sheet1.write(kk, 7, round(D8, 3))
    sheet1.write(kk, 8, round(D9, 3))
    sheet1.write(kk, 9, round(D10, 3))
    sheet1.write(kk, 10, round(D11, 3))
    sheet1.write(kk, 11, round(D12, 3))
    sheet1.write(kk, 12, round(D13, 3))
    sheet1.write(kk, 13, round(D14, 3))
    sheet1.write(kk, 14, round(D15, 3))
    sheet1.write(kk, 15, round(D16, 3))
    sheet1.write(kk, 16, round(D17, 3))

    file.save('result(meidu).xls')
