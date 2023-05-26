import onnxruntime
import numpy as np
import cv2
import matplotlib.pyplot as plt

#progress bar
import tqdm
#args parser
import argparse
parser = argparse.ArgumentParser(description='ONNX Super Resolution')
#model name
parser.add_argument('--model', type=str, default='RDN_model.onnx', help='model name')
#image name
parser.add_argument('--image', type=str, default='set100_1.jpeg', help='image name')
#upscaling factor
parser.add_argument('--scale', type=int, default=4, help='upscale factor')
#filtering options 0,1,2
parser.add_argument('--filter', type=int, default=0, help='filtering options 0,1,2')
#model input size
parser.add_argument('--input_size', type=int, default=64, help='model input size')

args = parser.parse_args()

def main():

    img_url = args.image
    img = cv2.imread(img_url)
    #img = img[:args.input_size, :args.input_size, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    img = np.transpose(img, (2, 0, 1))
    img = img / 255.0
    img = img.astype(np.float32)
    img = img[np.newaxis, :, :, :]
    print('Loaded image!, shape: {}\n'.format(img.shape[2:] ))

    #image padding to be divisible by input_size
    img = np.pad(img, ((0,0),(0,0),(0,args.input_size-img.shape[2]%args.input_size),(0,args.input_size-img.shape[3]%args.input_size)), 'reflect')

    print('Padded image, shape: {}\n'.format(img.shape[2:] ))


    #load onnx model
    sess = onnxruntime.InferenceSession(args.model)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    print('Up scaling image ...\n')
    shape = list(img.shape)
    shape[2] = shape[2] * args.scale
    shape[3] = shape[3] * args.scale



    output_image = np.zeros(shape)
    total_chunks = np.floor(img.shape[2]/args.input_size) * np.floor(img.shape[3]/args.input_size)

    counter = 0
    #progress bar
    t = tqdm.tqdm(total=total_chunks)

    for i in range(0,img.shape[2]-args.input_size+1,args.input_size):
        for j in range(0,img.shape[3]-args.input_size+1,args.input_size):
            img_chunk = img[:,:,i:i+args.input_size,j:j+args.input_size]
            #print('procesiing chunk: {}\n'.format(img_chunk.shape) )
            output = sess.run([output_name], {input_name: img_chunk})
        
            output_image[:,:,i*args.scale:i*args.scale+args.scale*args.input_size,j*args.scale:j*args.scale+args.scale*args.input_size] = output[0]
            #print ('output shape: {}\n'.format(output[0].shape))
            counter += 1
            #progress bar update
            t.update(1)



    t.close()


    out_img = output_image[0]
    print('Finished!, output shape: {}\n'.format(out_img.shape[1:]) )
    print('Saving image ...\n')

    out_img = np.transpose(out_img, (1, 2, 0))
    out_img = out_img * 255.0
    out_img = out_img.astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    if args.filter == 1:
        #gaussian filter
        out_img = cv2.GaussianBlur(out_img,(5,5),0)
        print('Gaussian filter applied!\n')
    elif args.filter == 2:
        #median filter
        out_img = cv2.medianBlur(out_img,5)
        print('Median filter applied!\n')

    output_image_name = "".join(args.image.split('.')[:-1]) + '_out.' + args.image.split('.')[-1]
    cv2.imwrite(output_image_name, out_img)
    print('Finished!\n')


if __name__ == '__main__':
    main()

