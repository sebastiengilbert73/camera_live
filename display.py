import cv2
import argparse
import logging
import os
from timeit import default_timer as timer
import ast
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main(
        outputDirectory,
        image_sizeHW,
        cameraID,
        preprocessing,
        capturesPeriod
):
    logging.info("display.main()")

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    capture = cv2.VideoCapture(cameraID)
    number_of_captures = 0
    start = timer()
    while True:
        ret_val, image = capture.read()
        if ret_val == True:

            if image_sizeHW is not None:
                image = cv2.resize(image, (image_sizeHW[1], image_sizeHW[0]))

            if preprocessing is not None:
                if preprocessing == 'grayscale':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif preprocessing == 'grayscale_blur3x3':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.blur(image, (3, 3))
                elif preprocessing == 'laplacian':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.Laplacian(image, ddepth=cv2.CV_8U, delta=128)
                    #image = cv2.blur(image, (3, 3))
                else:
                    raise NotImplementedError(
                        "display.main(): Not implemented preprocessing '{}'".format(preprocessing))
            # image.shape = (H, W, C)

            # Display the image
            cv2.imshow('image', image)

            number_of_captures += 1
            if number_of_captures == capturesPeriod:
                end = timer()
                delay_in_seconds = end - start
                fps = number_of_captures / delay_in_seconds
                logging.info("rate = {} fps".format(fps))
                start = timer()
                number_of_captures = 0
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # Save the image
            dateTime_obj = datetime.now()
            timestamp = dateTime_obj.strftime("%Y%m%d-%H:%M:%S")
            image_filepath = os.path.join(outputDirectory, timestamp) + ".png"
            logging.info(f"Saving {image_filepath}")
            cv2.imwrite(image_filepath, image)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory',
                        help="The output directory. Default: './outputs_display'",
                        default='./outputs_display')
    parser.add_argument('--imageSizeHW', help="The image resize (Height, Width), if desired. Default: 'None'",
                        default='None')
    parser.add_argument('--cameraID', help="The camera ID. Default: 0", type=int, default=0)
    parser.add_argument('--preprocessing', help="The preprocessing. Default: 'None'", default='None')
    parser.add_argument('--capturesPeriod', help="The number of captures used to compute frame rate. Defaut: 50", type=int, default=50)
    args = parser.parse_args()

    image_sizeHW = None
    if args.imageSizeHW.upper() != 'NONE':
        image_sizeHW = ast.literal_eval(args.imageSizeHW)
    preprocessing = None
    if args.preprocessing.upper() != 'NONE':
        preprocessing = args.preprocessing

    main(
        args.outputDirectory,
        image_sizeHW,
        args.cameraID,
        preprocessing,
        args.capturesPeriod
    )