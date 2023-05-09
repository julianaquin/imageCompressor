import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


image = Image.open('Galleta.jpg')

#convert image to black and white
grayimg = image.convert('LA')

#covert image to matrix
imgmat = np.array(list(grayimg.getdata(band=0)), float)
imgmat.shape = (grayimg.size[1], grayimg.size[0])
imgmat = np.matrix(imgmat)
plt.imshow(imgmat, cmap='gray')

originalSize = imgmat.shape[0]*imgmat.shape[1]
print('IMAGE OG SIZE', originalSize)

#calculate SVD of image
U, s, V = np.linalg.svd(imgmat)

i = 60
compressimg = np.matrix(U[:, :i]) * np.diag(s[:i]) * np.matrix(V[:i, :])

newSize = i*2489 + i + i*2207
print('COMPRESSED IMAGE', newSize)

if newSize < originalSize:
    print('COMPRESSION SUCCESSFULL')

#convert new matrix of compressed image to an actual image and save
image2 = Image.fromarray(compressimg)
image2 = image2.convert("L")
image2.save('COMPRESSED_IMG.jpg')


plt.imshow(compressimg, cmap='gray')
plt.title('Galleta')
plt.show()
