import numpy
import re
import sys
import scipy.misc
import subprocess
import os.path
import imageio

import vifp
import ssim
import psnr
import niqe
import reco


import matplotlib.pyplot as plt

ref_file = 'demo/lena.png'
dist_file = 'demo/lena_tmp.jpg'

ref = imageio.imread(ref_file,as_gray=True).astype(numpy.float32)

quality_values = []
size_values = []
vifp_values = []
ssim_values = []
psnr_values = []
niqe_values = []
reco_values = []

plt.figure(figsize=(8, 8))

quality=101

# subprocess.check_call('gm convert %s -quality %d %s'%(ref_file, quality, dist_file), shell=True)
file_size = os.path.getsize(dist_file)

# dist = imageio.imread(dist_file, flatten=True).astype(numpy.float32)
dist = imageio.imread(dist_file,as_gray=True).astype(numpy.float32)

quality_values.append( quality )
size_values.append( int(file_size/1024) )
vifp_values.append( vifp.vifp_mscale(ref, dist) )
ssim_values.append( ssim.ssim_exact(ref/255, dist/255) )
psnr_values.append( psnr.psnr(ref, dist) )
niqe_values.append( niqe.niqe(dist/255) )
reco_values.append( reco.reco(ref/255, dist/255) )



print(size_values)
print(vifp_values)
print(ssim_values)
print(psnr_values)
print(niqe_values)
print(reco_values)
