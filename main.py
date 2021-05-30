import cv2 as cv
import numpy as np


haystack_img = cv.imread('gae1.jpg', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('alert.jpg', cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

print('Best Match top left positopn: %s' % str(max_loc))
print('Best Match cinfidence: %s' % max_val)

threshold = 0.8
if max_val >= threshold:
    # cv.imshow('Original', haystack_img)
    print('Found needle.')

    needle_w = needle_img.shape[1]
    needle_h = needle_img.shape[0]

    top_left = max_loc
    bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

    cv.rectangle(haystack_img, top_left, bottom_right, color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)

    cv.imshow('Result', haystack_img)
    cv.waitKey()
    # cv.imwrite('result1.jpg', haystack_img)
else:
    print('Needle not found.')

# cv.imshow('Originnal', result)
# cv.waitKey()