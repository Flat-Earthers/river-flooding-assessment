from snappy import ProductIO, GPF, jpy
import numpy as np
import matplotlib.pyplot as plt
import time

ProductIO.readProduct('/home/andrea/flat-earthers/S2B_MSIL1C_20200210T031849_N0209_R118_T50TLS_20200210T061800.zip')


def s2_resampler(product, resolution=20, upmethod='Bilinear', downmethod='Mean',
                 flag='FlagMedianAnd', opt=False):
    '''Resampler operator dedicated to Sentinel2-msi characteristics (e.g., viewing angles)
    :param product: S2-msi product as provided by esasnappy ProductIO.readProduct()
    :param resolution: target resolution in meters (10, 20, 60)
    :param upmethod: interpolation method ('Nearest', 'Bilinear', 'Bicubic')
    :param downmethod: aggregation method ('First', 'Min', 'Max', 'Mean', 'Median')
    :param flag: method for flags aggregation ('First', 'FlagAnd', 'FlagOr', 'FlagMedianAnd', 'FlagMedianOr')
    :param opt: resample on pyramid levels (True/False)
    :return: interpolated target product'''

    res = str(resolution)

    resampler = jpy.get_type('org.esa.s2tbx.s2msi.resampler.S2ResamplingOp')

    op = resampler()
    op.setParameter('targetResolution', res)
    op.setParameter('upsampling', upmethod)
    op.setParameter('downsampling', downmethod)
    op.setParameter('flagDownsampling', flag)
    op.setParameter('resampleOnPyramidLevels', opt)
    op.setSourceProduct(product)

    return op.getTargetProduct()


def c2rcc_params():
    GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
    op_spi = GPF.getDefaultInstance().getOperatorSpiRegistry().getOperatorSpi('c2rcc.msi')
    print(dir(op_spi))
    print('Op name:', op_spi.getOperatorDescriptor().getName())
    print('Op alias:', op_spi.getOperatorDescriptor().getAlias())
    param_Desc = op_spi.getOperatorDescriptor().getParameterDescriptors()
    for param in param_Desc:
        print(param.getName(), "or", param.getAlias())

def c2rcc(product):
    GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()
    op_spi = GPF.getDefaultInstance().getOperatorSpiRegistry().getOperatorSpi('c2rcc.msi')
    c2rcc_operator = op_spi.createOperator()
    c2rcc_operator.setSourceProduct(product)
    c2rcc_operator.setParameter('validPixelExpression', "B8 > 0 && B8 < 0.12")
    c2rcc_operator.setParameter('salinity', "20")
    c2rcc_operator.setParameter('temperature', "15")
    c2rcc_operator.setParameter('ozone', "330")
    c2rcc_operator.setParameter('press', "1000")
    c2rcc_operator.setParameter('elevation', "0")
    c2rcc_operator.setParameter('TSMfakBpart', "1.72")
    c2rcc_operator.setParameter('TSMfakBwit', "3.1")
    c2rcc_operator.setParameter('CHLexp', "1.04")
    c2rcc_operator.setParameter('CHLfak', "21.0")
    c2rcc_operator.setParameter('thresholdRtosaOOS', "0.05")
    c2rcc_operator.setParameter('thresholdAcReflecOos', "0.1")
    c2rcc_operator.setParameter('thresholdCloudTDown865', "0.955")
    c2rcc_operator.setParameter('netSet', "C2RCC-Nets")

    c2rcc_operator.setParameter('outputAsRrs', "False")
    c2rcc_operator.setParameter('deriveRwFromPathAndTransmittance', "False")
    c2rcc_operator.setParameter('outputRtoa', "True")
    c2rcc_operator.setParameter('outputRtosaGc', "False")
    c2rcc_operator.setParameter('outputRtosaGcAann', "False")
    c2rcc_operator.setParameter('outputRpath', "False")
    c2rcc_operator.setParameter('outputTdown', "False")
    c2rcc_operator.setParameter('outputTup', "False")
    c2rcc_operator.setParameter('outputAcReflectance', "True")
    c2rcc_operator.setParameter('outputRhown', "True")
    c2rcc_operator.setParameter('outputOos', "False")
    c2rcc_operator.setParameter('outputKd', "True")
    c2rcc_operator.setParameter('outputUncertainties', "True")

    c2rcc_operator.setSourceProduct(product)
    return c2rcc_operator.getTargetProduct()


def print_pd(result, band, filename):
    rad13 = result.getBand(band)
    w = rad13.getRasterWidth()
    h = rad13.getRasterHeight()
    rad13_data = np.zeros(w * h, np.float32)
    rad13.readPixels(0, 0, w, h, rad13_data)
    # p.dispose()
    rad13_data.shape = h, w
    imgplot = plt.imshow(rad13_data)
    imgplot.write_png(filename)



file_path = '/home/andrea/flat-earthers/S2B_MSIL1C_20200210T031849_N0209_R118_T50TLS_20200210T061800.zip'

input_file = ProductIO.readProduct(file_path)

# c2rcc_params()

start = time.time()
resampled = s2_resampler(input_file, 10)
print("Took ", time.time()-start)

start = time.time()
c2rcc_result = c2rcc(resampled)
print("Took ", time.time()-start)

print_pd(c2rcc_result, 'unc_agelb', 'pd.png')