from numpy import fft
import numpy
from scipy import optimize


def move(image, x, y):
    imagesize = image.shape[0]
    cent = numpy.where(image == numpy.max(image))
    dx = int(numpy.max(cent[1]) - x)
    dy = int(numpy.max(cent[0]) - y)
    if dy > 0:
        if dx > 0:
            arr_y = image[0:dy, 0:imagesize]
            arr = image[dy:imagesize, 0:imagesize]
            arr = numpy.row_stack((arr, arr_y))
            arr_x = arr[0:imagesize, 0:dx]
            arr = arr[0:imagesize, dx:imagesize]
            arr = numpy.column_stack((arr, arr_x))
            return arr
        elif dx == 0:
            arr_y = image[0:dy, 0:imagesize]
            arr = image[dy:imagesize, 0:imagesize]
            arr = numpy.row_stack((arr, arr_y))
            return arr
        else:
            arr_y = image[0:dy, 0:imagesize]
            arr = image[dy:imagesize, 0:imagesize]
            arr = numpy.row_stack((arr, arr_y))
            arr_x = arr[0:imagesize, 0:imagesize + dx]
            arr = arr[0:imagesize, imagesize + dx:imagesize]
            arr = numpy.column_stack((arr, arr_x))
            return arr
    elif dy == 0:
        if dx > 0:
            arr_x = image[0:imagesize, 0:dx]
            arr = image[0:imagesize, dx:imagesize]
            arr = numpy.column_stack((arr, arr_x))
            return arr
        elif dx == 0:
            return image
        else:
            arr = image[0:imagesize, 0:imagesize + dx]
            arr_x = image[0:imagesize, imagesize + dx:imagesize]
            arr = numpy.column_stack((arr_x, arr))
            return arr
    elif dy < 0:
        if dx > 0:
            arr_y = image[imagesize + dy:imagesize, 0:imagesize]
            arr = image[0:imagesize + dy, 0:imagesize]
            arr = numpy.row_stack((arr_y, arr))
            arr_x = arr[0:imagesize, 0:dx]
            arr = arr[0:imagesize, dx:imagesize]
            arr = numpy.column_stack((arr, arr_x))
            return arr
        elif dx == 0:
            arr_y = image[imagesize + dy:imagesize, 0:imagesize]
            arr = image[0:imagesize + dy, 0:imagesize]
            arr = numpy.row_stack((arr_y, arr))
            return arr
        else:
            arr_y = image[imagesize + dy:imagesize, 0:imagesize]
            arr = image[0:imagesize + dy, 0:imagesize]
            arr = numpy.row_stack((arr_y, arr))
            arr_x = arr[0:imagesize, 0:imagesize + dx]
            arr = arr[0:imagesize, imagesize + dx:imagesize]
            arr = numpy.column_stack((arr, arr_x))
            return arr

def psf_align(image):
    imagesize = image.shape[0]
    arr = move(image, 0, 0)
    image_f = fft.fft2(arr)
    yy, xx = numpy.mgrid[0:48, 0:48]
    xx = numpy.mod(xx + 24, 48) - 24
    yy = numpy.mod(yy + 24, 48) - 24
    fk = numpy.abs(image_f) ** 2
    line = numpy.sort(numpy.array(fk.flat))
    idx = fk < 4 * line[int(imagesize ** 2 / 2)]
    fk[idx] = 0
    weight = fk / line[int(imagesize ** 2 / 2)]
    kx = xx * 2 * numpy.pi / imagesize
    ky = yy * 2 * numpy.pi / imagesize

    def pha(p):
        x, y = p
        return numpy.sum((numpy.angle(image_f * numpy.exp(-1.0j * (kx * x + ky * y)))) ** 2 * weight)

    res = fmin_cg(pha, [0, 0], disp=False)
    inve = fft.fftshift(numpy.real(fft.ifft2(image_f * numpy.exp(-1.0j * (kx * res[0] + ky * res[1])))))
    return inve