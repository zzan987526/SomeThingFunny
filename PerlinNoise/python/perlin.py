# -*- coding:utf-8 -*-
import numpy as np
import random, math
from PIL import Image
import numpy as np
from numpy.ctypeslib import ndpointer
import matplotlib.pyplot as plt
from ctypes import *
from time import time


"""
Texture generation using Perlin noise
"""
class NoiseUtils:

    def __init__(self, imageSize):
        self.imageSize = imageSize
        self.gradientNumber = 256

        self.grid = [[]]
        self.gradients = []
        self.permutations = []
        self.img = {}

        self.__generateGradientVectors()
        self.__normalizeGradientVectors()
        self.__generatePermutationsTable()

    def __generateGradientVectors(self):
        for i in range(self.gradientNumber):
            while True:
                x, y = random.uniform(-1, 1), random.uniform(-1, 1)
                if x * x + y * y < 1:
                    self.gradients.append([x, y])
                    break

    def __normalizeGradientVectors(self):
        for i in range(self.gradientNumber):
            x, y = self.gradients[i][0], self.gradients[i][1]
            length = math.sqrt(x * x + y * y)
            self.gradients[i] = [x / length, y / length]

    # The modern version of the Fisher-Yates shuffle
    def __generatePermutationsTable(self):
        self.permutations = [i for i in range(self.gradientNumber)]
        for i in reversed(range(self.gradientNumber)):
            j = random.randint(0, i)
            self.permutations[i], self.permutations[j] = \
                self.permutations[j], self.permutations[i]

    def getGradientIndex(self, x, y):
        return self.permutations[(x + self.permutations[y % self.gradientNumber]) % self.gradientNumber]

    def perlinNoise(self, x, y):
        qx0 = int(math.floor(x))
        qx1 = qx0 + 1

        qy0 = int(math.floor(y))
        qy1 = qy0 + 1

        q00 = self.getGradientIndex(qx0, qy0)
        q01 = self.getGradientIndex(qx1, qy0)
        q10 = self.getGradientIndex(qx0, qy1)
        q11 = self.getGradientIndex(qx1, qy1)

        tx0 = x - math.floor(x)
        tx1 = tx0 - 1

        ty0 = y - math.floor(y)
        ty1 = ty0 - 1

        v00 = self.gradients[q00][0] * tx0 + self.gradients[q00][1] * ty0
        v01 = self.gradients[q01][0] * tx1 + self.gradients[q01][1] * ty0
        v10 = self.gradients[q10][0] * tx0 + self.gradients[q10][1] * ty1
        v11 = self.gradients[q11][0] * tx1 + self.gradients[q11][1] * ty1

        wx = tx0 * tx0 * (3 - 2 * tx0)
        v0 = v00 + wx * (v01 - v00)
        v1 = v10 + wx * (v11 - v10)

        wy = ty0 * ty0 * (3 - 2 * ty0)
        return (v0 + wy * (v1 - v0)) * 0.5 + 1

    def makeTexture(self, texture = None):
        if texture is None:
            texture = self.cloud

        noise = {}
        max = min = None
        for i in range(self.imageSize):
            for j in range(self.imageSize):
                value = texture(i, j)
                noise[i, j] = value

                if max is None or max < value:
                    max = value

                if min is None or min > value:
                    min = value

        for i in range(self.imageSize):
            for j in range(self.imageSize):
                self.img[i, j] = (int) ((noise[i, j] - min) / (max - min) * 255 )

    def fractalBrownianMotion(self, x, y, func):
        octaves = 12
        amplitude = 1.0
        frequency = 1.0 / self.imageSize
        persistence = 0.5
        value = 0.0
        for k in range(octaves):
            value += func(x * frequency, y * frequency) * amplitude
            frequency *= 2
            amplitude *= persistence
        return value

    def cloud(self, x, y, func = None):
        if func is None:
            func = self.perlinNoise

        return self.fractalBrownianMotion(8 * x, 8 * y, func)

    def wood(self, x, y, noise = None):
        if noise is None:
            noise = self.perlinNoise

        frequency = 1.0 / self.imageSize
        n = noise(4 * x * frequency, 4 * y * frequency) * 10
        return n - int(n)

    def marble(self, x, y, noise = None):
        if noise is None:
            noise = self.perlinNoise

        frequency = 1.0 / self.imageSize
        n = self.fractalBrownianMotion(8 * x, 8 * y, self.perlinNoise)
        return (math.sin(16 * x * frequency + 4 * (n - 0.5)) + 1) * 0.5

# if __name__ == "__main__":
#     imageSize = 256
#     noise = NoiseUtils(imageSize)
#     noise.makeTexture(texture = noise.cloud)

#     img = np.zeros((imageSize, imageSize))

#     pixels = img.copy()

#     for i in range(0, imageSize):
#        for j in range(0, imageSize):
#             c = noise.img[i, j]
#             pixels[i, j] = c
#     print pixels[100,100]
#     pixels = pixels.astype('uint8')
#     print pixels[100,100]
#     im = Image.fromarray(pixels)
#     im.show()
#     #cv2.imwrite('Noise.png', pixels) 

# --------------------------------------------------------------------------------------
# 3d implement
permutation = [ 151,160,137,91,90,15,
131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
]
p=permutation*2

def PerlinNoise(x,y,z, octaves=6, persistence=0.5):
    # Sum of Noise Function = Perlin Noise
    # Each successive noise function you add is known as an octave
    total = 0
    p = persistence  # reference value:  1/4, 1/2 ,3/4 
    for i in range(octaves):
        frequency = 2**i
        amplitude=p**i
        octave=ImprovedNoise(x * frequency, y * frequency, z * frequency) * amplitude
        total+=octave
    return total

def ImprovedNoise(x, y, z):
	
    # frequency=1/wavelength
    # It returns floating point numbers between -1.0 and 1.0
    # FIND UNIT CUBE THAT CONTAINS POINT.
    X = int(math.floor(x)) & 255
    Y = int(math.floor(y)) & 255
    Z = int(math.floor(z)) & 255
    # FIND RELATIVE X,Y,Z OF POINT IN CUBE.
    x -= math.floor(x)
    y -= math.floor(y)
    z -= math.floor(z)

    # COMPUTE FADE CURVES FOR EACH OF X,Y,Z.
    u,v,w = fade(x),fade(y),fade(z)

    # HASH COORDINATES OF THE 8 CUBE CORNERS
    # AND ADD BLENDED RESULTS FROM  8 CORNERS OF CUBE
    global p
    A = p[X]+Y
    AA = p[A]+Z
    AB = p[A+1]+Z
    B = p[X+1]+Y
    BA = p[B]+Z
    BB = p[B+1]+Z
    return lerp(w, lerp(v, lerp(u, grad(p[AA  ], x  , y  , z   ),
                                   grad(p[BA  ], x-1, y  , z   )),
                           lerp(u, grad(p[AB  ], x  , y-1, z   ),
                                   grad(p[BB  ], x-1, y-1, z   ))),
                   lerp(v, lerp(u, grad(p[AA+1], x  , y  , z-1 ),
                                   grad(p[BA+1], x-1, y  , z-1 )),
                           lerp(u, grad(p[AB+1], x  , y-1, z-1 ),
                                   grad(p[BB+1], x-1, y-1, z-1 ))))

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(t, a, b):
    return a + t * (b - a)

def grad(hash, x, y, z):
    # CONVERT LO 4 BITS OF HASH CODE INTO 12 GRADIENT DIRECTIONS.
    h = hash & 15
    u = x if h < 8 else y
    v = y if h < 4 else x if h==12 or h==14 else z

    return (u if (h&1)==0 else -u)+(v if (h&2)==0 else -v)



# data = []
# size = 256
# scale = 50.0
# t1 = time()
# for i in range(size):
# 	for j in range(size):
# 		data.append(PerlinNoise(i/scale, j/scale, 0))
# print time() - t1

# im = Image.new("L", (size, size))
# im.putdata(data, 128, 128)  # 此处128不是图片size
# im.show()

# ------------------------------------------------------------------------
# 2d implement
perm = range(256)  # 随机排列
random.shuffle(perm)
perm += perm  # 使其有周期
dirs = [(math.cos(a * 2.0 * math.pi / 256),  # X方向cos，Y方向sin 插值
         math.sin(a * 2.0 * math.pi / 256))
         for a in range(256)]
 
 
def noise(x, y, per):
    # Perlin noise is generated from a summation of little "surflets" which are the product of a randomly oriented
    # gradient and a separable polynomial falloff function.
    def surflet(gridX, gridY):  # gridX, gridY 为晶体格顶点的坐标
        distX, distY = abs(x-gridX), abs(y-gridY)  # 距离向量
        polyX = 1 - 6*distX**5 + 15*distX**4 - 10*distX**3  # polynomial falloff function
        polyY = 1 - 6*distY**5 + 15*distY**4 - 10*distY**3
        # 此处hash函数用于每个输入坐标都有一个对应的排列值
        hashed = perm[perm[int(gridX) % per] + int(gridY) % per]
        grad = (x-gridX)*dirs[hashed][0] + (y-gridY)*dirs[hashed][1]
        return polyX * polyY * grad
    intX, intY = int(x), int(y)
    # 4个方向结果的累加，图中黄色和蓝色面积的累加。4个晶体格。
    return (surflet(intX+0, intY+0) + surflet(intX+1, intY+0) +
            surflet(intX+0, intY+1) + surflet(intX+1, intY+1))

def fBm(x, y, per, octs):
    val = 0
    for o in range(octs):
        val += 0.5**o * noise(x*2**o, y*2**o, per*2**o)
    return val
 
 
# size, freq, octs, data = 256, 1/32.0, 5, []
# t1 = time()
# for y in range(size):
#     for x in range(size):
#         data.append(fBm(x*freq, y*freq, int(size*freq), octs))
# print time() - t1
# im = Image.new("L", (size, size))
# im.putdata(data, 128, 128)  # 此处128不是图片size
# im.show()

# ------------------------------------------------------------------------------------
# numpy mat oper implement
def generate_perlin_noise_3d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1],0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    # Gradients
    theta = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    phi = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    gradients = np.stack((np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)), axis=3)
    g000 = gradients[0:-1,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g100 = gradients[1:  ,0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g010 = gradients[0:-1,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g110 = gradients[1:  ,1:  ,0:-1].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g001 = gradients[0:-1,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g101 = gradients[1:  ,0:-1,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g011 = gradients[0:-1,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g111 = gradients[1:  ,1:  ,1:  ].repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    # Ramps
    n000 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g000, 3)
    n100 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g100, 3)
    n010 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g010, 3)
    n110 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g110, 3)
    n001 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g001, 3)
    n101 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g101, 3)
    n011 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g011, 3)
    n111 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g111, 3)
    # Interpolation
    t = f(grid)
    n00 = n000*(1-t[:,:,:,0]) + t[:,:,:,0]*n100
    n10 = n010*(1-t[:,:,:,0]) + t[:,:,:,0]*n110
    n01 = n001*(1-t[:,:,:,0]) + t[:,:,:,0]*n101
    n11 = n011*(1-t[:,:,:,0]) + t[:,:,:,0]*n111
    n0 = (1-t[:,:,:,1])*n00 + t[:,:,:,1]*n10
    n1 = (1-t[:,:,:,1])*n01 + t[:,:,:,1]*n11
    return ((1-t[:,:,:,2])*n0 + t[:,:,:,2]*n1)
    
def generate_fractal_noise_3d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_3d(shape, (frequency*res[0], frequency*res[1], frequency*res[2]))
        frequency *= 2
        amplitude *= persistence
    return noise
    
# if __name__ == '__main__':
#     import matplotlib.animation as animation
    
#     np.random.seed(0)
#     noise = generate_fractal_noise_3d((32, 256, 256), (1, 4, 4), 4)
    
#     fig = plt.figure()
#     images = [[plt.imshow(layer, cmap='gray', interpolation='lanczos', animated=True)] for layer in noise]
#     animation = animation.ArtistAnimation(fig, images, interval=50, blit=True)
#     plt.show()

#--------------------------------------------------------------------------------------
# C++ implement
def octavePerlin2d(lattice, res, octaves = 1, persistence=0.5):

	libc = CDLL("PerlinNoise.dll")
	# get the 2d Perlin noise function
	perlinNoise2D = libc.perlinNoise2D
	# Need specify the types of the argument for function perlinNoise2D
	perlinNoise2D.argtypes = (c_int, c_int, 
                                  c_int, c_int)

	# This note is extremely useful to understand how to return a 2d array!
	# https://stackoverflow.com/questions/43013870/
        # how-to-make-c-return-2d-array-to-python?noredirect=1&lq=1
	# We can never pass a 2d array, therefore return 1d array in a C function
	#perlinNoise2D.restype = POINTER(c_float)
	perlinNoise2D.restype = ndpointer(dtype=c_float, 
                                          shape = (res[0], res[1]))
	noises = np.zeros(res)
	frequency = 1
	amplitude = 1
	for _ in range(octaves):
		temp = perlinNoise2D(c_int(frequency*lattice[1]), 
			c_int(frequency*lattice[0]), 
			c_int(res[1]), 
			c_int(res[0]) )
		noises += amplitude * temp
		frequency *= 2
		amplitude *= persistence
	return noises

t1 = time()
result = octavePerlin2d((8,8), (512, 512), 6)
print (time()-t1)

plt.imshow(result, cmap = 'gray')
plt.tight_layout()
plt.savefig("noise2d.png")
plt.show()