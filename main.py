from processImageOCV import processImage
import time
import subprocess

file = 'license4processed.png'
param = 7

tic = time.clock()
processImage(file='camion2.png', output='license2processed.png')
subprocess.call(['tesseract', 'license2processed.png', 'stdout', '-psm', str(param)])

processImage(file='patente5.png', output='license5processed.png')
subprocess.call(['tesseract', 'license5processed.png', 'stdout', '-psm', str(param)])

processImage(file='patente4.png', output='license4processed.png')
subprocess.call(['tesseract', 'license4processed.png', 'stdout', '-psm', str(param)])


toc = time.clock()

print('Time: %f' % (toc - tic))
