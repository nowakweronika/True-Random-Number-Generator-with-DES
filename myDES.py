from PIL import Image, ImageOps
from numpy import *
from scipy.stats import entropy as esp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

###################################################################################################
########################          TRNG VARIABLES DESCRIPTION              #########################
###################################################################################################
# png - lena image
# gray_png - gray lena
# imageConvert - lena dithering img
# arnoldArray - array with arnold
# arnoldArray1D - 1D array with arnold


###################################################################################################
########################          OPERATIONS WITH IMAGE                   #########################
###################################################################################################
#load image
png = Image.open('lena.png')

#create grey lena
gray_png = ImageOps.grayscale(png)
gray_png.save('lenaGray.png')

#lena dithering
imageConvert = Image.open('lenaGray.png').convert(mode='1',dither=Image.FLOYDSTEINBERG)
imageConvert.save('lenaDit.png')

#load lena dithering img
im = array(Image.open('lenaDit.png'))
N = im.shape[0]


###################################################################################################
########################          ARNOLDS CAT MAP                         #########################
###################################################################################################
#parameter p and q
p = 1
q = 1

#create x and y components of Arnold's cat mapping
x,y = meshgrid(range(N),range(N))
xmap = (x + (p * y)) % N
ymap = ((q * x) + (q * p * y) + y) % N

#create arnolds cat map
for i in range(7):
	result = Image.fromarray(im)
	im = im[xmap,ymap]
result.save("arnold.png")

#load image
imageArnold = Image.open('arnold.png')
arnoldArray = np.array(imageArnold)

#size of arnolds array
rows = len(arnoldArray)
columns = len(arnoldArray[0])
totalLength = rows * columns

#1D array, where arnolds data will be stored
arnoldArray1D = [0] * totalLength

#1D array with arnold bits after dithering
helper = 0
for i in range(rows):
	for j in range(columns):
		arnoldArray1D[helper] = arnoldArray[i][j]
		helper += 1

#convert boolean to string array
ArnoldData = arnoldArray1D
for i in range(len(ArnoldData)):
	if(ArnoldData[i] == True):
		ArnoldData[i] = '1'
	else:
		ArnoldData[i] = '0'


###################################################################################################
########################          DES FUNCTIONS                           #########################
###################################################################################################
def hex2bin(s):
    mp = {'0' : "0000", '1' : "0001", '2' : "0010", '3' : "0011",
          '4' : "0100", '5' : "0101", '6' : "0110", '7' : "0111",
          '8' : "1000", '9' : "1001", 'A' : "1010", 'B' : "1011",
          'C' : "1100", 'D' : "1101", 'E' : "1110", 'F' : "1111" }
    bin = ""
    for i in range(len(s)):
        bin = bin + mp[s[i]]
    return bin

def bin2hex(s):
    mp = {"0000" : '0', "0001" : '1', "0010" : '2', "0011" : '3',
          "0100" : '4', "0101" : '5', "0110" : '6', "0111" : '7', 
          "1000" : '8', "1001" : '9', "1010" : 'A', "1011" : 'B', 
          "1100" : 'C', "1101" : 'D', "1110" : 'E', "1111" : 'F' }
    hex = ""
    for i in range(0,len(s),4):
        ch = ""
        ch = ch + s[i]
        ch = ch + s[i + 1] 
        ch = ch + s[i + 2] 
        ch = ch + s[i + 3] 
        hex = hex + mp[ch] 
    return hex

def bin2dec(bin):
    dec = 0
    i = len(bin)
    j = 0
    while(i):
        dec += int(bin[i-1])*(2**j)
        i -= 1
        j += 1
    return dec

def dec2bin(dec):
    bin = ""
    i = 4
    while(i):
        if(dec - 2**(i-1) < 0):
            bin += "0"
        else:
            bin += "1"
            dec -= 2**(i-1)
        i -= 1
    return bin

def sboxFunction(chain, n):
    first = chain[0]
    last = chain[len(chain)-1]
    i = bin2dec(first + last)
    j = ""
    for k in range(1, len(chain)-1):
        j += chain[k]
    j = bin2dec(j)
    return sbox[n][i][j]

def permutation(myKey, permKey):
    newKey = ""
    for i in range(len(permKey)):
        newKey += myKey[permKey[i] - 1]
    return newKey

def leftShiftFun(chain, shift):
    helper = [""] * 2
    newChain = ""
    for i in range(shift):
        helper[i] += chain[i]
    for j in range(len(chain) - shift):
        newChain += chain[j + shift]
    newChain += helper[0]
    newChain += helper[1]
    return newChain

def xor(k, er):
    xorChain = ""
    for i in range(len(k)):
        if k[i] == er[i]:
            xorChain += "0"
        else:
            xorChain += "1"
    return xorChain


###################################################################################################
########################          DES TABLES                              #########################
###################################################################################################
PC1 = [57, 49, 41, 33, 25, 17,  9, 
        1, 58, 50, 42, 34, 26, 18, 
       10,  2, 59, 51, 43, 35, 27, 
       19, 11,  3, 60, 52, 44, 36, 
       63, 55, 47, 39, 31, 23, 15, 
        7, 62, 54, 46, 38, 30, 22, 
       14,  6, 61, 53, 45, 37, 29, 
       21, 13,  5, 28, 20, 12,  4]

leftShift = [1, 1, 2, 2, 
             2, 2, 2, 2, 
             1, 2, 2, 2, 
             2, 2, 2, 1]

PC2 = [14, 17, 11, 24,  1,  5, 
        3, 28, 15,  6, 21, 10, 
       23, 19, 12,  4, 26,  8, 
       16,  7, 27, 20, 13,  2, 
       41, 52, 31, 37, 47, 55, 
       30, 40, 51, 45, 33, 48, 
       44, 49, 39, 56, 34, 53, 
       46, 42, 50, 36, 29, 32]

IP = [58, 50, 42, 34, 26, 18, 10, 2, 
      60, 52, 44, 36, 28, 20, 12, 4, 
      62, 54, 46, 38, 30, 22, 14, 6, 
      64, 56, 48, 40, 32, 24, 16, 8, 
      57, 49, 41, 33, 25, 17,  9, 1, 
      59, 51, 43, 35, 27, 19, 11, 3, 
      61, 53, 45, 37, 29, 21, 13, 5, 
      63, 55, 47, 39, 31, 23, 15, 7]

eBitSelection = [32,  1,  2,  3,  4,  5,
                  4,  5,  6,  7,  8,  9,
                  8,  9, 10, 11, 12, 13,
                 12, 13, 14, 15, 16, 17, 
                 16, 17, 18, 19, 20, 21,
                 20, 21, 22, 23, 24, 25,
                 24, 25, 26, 27, 28, 29,
                 28, 29, 30, 31, 32,  1]

sbox =  [[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7], 
          [ 0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8], 
          [ 4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0], 
          [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13 ]],
             
         [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10], 
            [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5], 
            [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15], 
           [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9 ]], 
    
         [ [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8], 
           [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1], 
           [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7], 
            [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12 ]], 
        
          [ [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15], 
           [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9], 
           [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4], 
            [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14] ], 
         
          [ [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9], 
           [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6], 
            [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14], 
           [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3 ]], 
        
         [ [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11], 
           [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8], 
            [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6], 
            [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13] ], 
          
          [ [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1], 
           [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6], 
            [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2], 
            [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12] ], 
         
         [ [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7], 
            [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2], 
            [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8], 
            [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11] ] ]

P = [16,  7, 20, 21,
     29, 12, 28, 17,
      1, 15, 23, 26,
      5, 18, 31, 10,
      2,  8, 24, 14,
     32, 27,  3,  9,
     19, 13, 30,  6,
     22, 11,  4, 25]

IP1 = [40,  8, 48, 16, 56, 24, 64, 32,
       39,  7, 47, 15, 55, 23, 63, 31,
       38,  6, 46, 14, 54, 22, 62, 30,
       37,  5, 45, 13, 53, 21, 61, 29,
       36,  4, 44, 12, 52, 20, 60, 28,
       35,  3, 43, 11, 51, 19, 59, 27,
       34,  2, 42, 10, 50, 18, 58, 26,
       33,  1, 41,  9, 49, 17, 57, 25]


###################################################################################################
########################          DES ENCRYPTER                           #########################
###################################################################################################
lowerIndex = 0
higherIndex = 64
key = ""
myRange = int(len(ArnoldData)/64)

message = "0123456789ABCDEF"
print("Message is ", message)
for it in range(myRange):
    for jk in range(lowerIndex, higherIndex):
        key += ArnoldData[jk]
    lowerIndex += 64
    higherIndex += 64
    message = hex2bin(message)

    n = 17
    C = [""] * n #left half of permuted key
    D = [""] * n #right half of permuted key
    K = [""] * n #permuted keys
    L = [""] * n #left half of permuted message
    R = [""] * n #right half of permuted message
    B = [[] * 9 for i in range(n)] #6-bits strings from K(n)+E[R(n-1)]
    outSbox = [[] * 9 for i in range(n)] #output of the 8 sboxes
    permutedOutSbox = [""] * n

    #Step 1: Create 16 subkeys, each of which is 48-bits long.
    K[0] = permutation(key, PC1)
    C[0] = K[0][:28]
    D[0] = K[0][28:]

    for i in range(1, n):
        C[i] = leftShiftFun(C[i-1], leftShift[i-1])
        D[i] = leftShiftFun(D[i-1], leftShift[i-1])
        K[i] = C[i] + D[i]
        K[i] = permutation(K[i], PC2)

    #Step 2: Encode each 64-bit block of data.
    message = permutation(message, IP)
    L[0] = message[:32]
    R[0] = message[32:]

    for i in range(1, n):
        L[i] = R[i - 1]
        R[i] = xor(K[i], permutation(R[i - 1], eBitSelection))
        for j in range(8):
            B[i].append(R[i][(j+1)*6-6:(j+1)*6])
            outSbox[i].append(sboxFunction(B[i][j], j))
            outSbox[i][j] = dec2bin(outSbox[i][j])
            permutedOutSbox[i] += outSbox[i][j]
        permutedOutSbox[i] = permutation(permutedOutSbox[i], P)
        R[i] = xor(L[i-1], permutedOutSbox[i])

    RL = R[16] + L[16]
    RL = permutation(RL, IP1)
    RL = bin2hex(RL)
    print(it, " encrypted message is ", RL)
    key = ""

print("Due to the nature of my TRNG, from one image I can make 4096 keys.")
print("\n")

print("Example from http://page.math.tu-berlin.de/~kant/teaching/hess/krypto-ws2006/des.htm :")
key = "133457799BBCDFF1"
key = hex2bin(key)
message = "0123456789ABCDEF"
print("Message is ", message)
message = hex2bin(message)

n = 17
C = [""] * n #left half of permuted key
D = [""] * n #right half of permuted key
K = [""] * n #permuted keys
L = [""] * n #left half of permuted message
R = [""] * n #right half of permuted message
B = [[] * 9 for i in range(n)] #6-bits strings from K(n)+E[R(n-1)]
outSbox = [[] * 9 for i in range(n)] #output of the 8 sboxes
permutedOutSbox = [""] * n

#Step 1: Create 16 subkeys, each of which is 48-bits long.
K[0] = permutation(key, PC1)
C[0] = K[0][:28]
D[0] = K[0][28:]

for i in range(1, n):
    C[i] = leftShiftFun(C[i-1], leftShift[i-1])
    D[i] = leftShiftFun(D[i-1], leftShift[i-1])
    K[i] = C[i] + D[i]
    K[i] = permutation(K[i], PC2)

#Step 2: Encode each 64-bit block of data.
message = permutation(message, IP)
L[0] = message[:32]
R[0] = message[32:]

for i in range(1, n):
    L[i] = R[i - 1]
    R[i] = xor(K[i], permutation(R[i - 1], eBitSelection))
    for j in range(8):
        B[i].append(R[i][(j+1)*6-6:(j+1)*6])
        outSbox[i].append(sboxFunction(B[i][j], j))
        outSbox[i][j] = dec2bin(outSbox[i][j])
        permutedOutSbox[i] += outSbox[i][j]
    permutedOutSbox[i] = permutation(permutedOutSbox[i], P)
    R[i] = xor(L[i-1], permutedOutSbox[i])

RL = R[16] + L[16]
RL = permutation(RL, IP1)
RL = bin2hex(RL)
print("Encrypted message is ", RL)
print("\n\n")