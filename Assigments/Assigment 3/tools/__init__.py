import numpy as np
import pickle
import os


def collage(data):
    if type(data) is not list:
        if data.shape[3] != 3:
            data = data.transpose(0, 2, 3, 1)

        images = [img for img in data]
    else:
        images = list(data)

    side = int(np.ceil(len(images)**0.5))
    for i in range(side**2 - len(images)):
        images.append(images[-1])
    collage = [np.concatenate(images[i::side], axis=0)
               for i in range(side)]
    collage = np.concatenate(collage, axis=1)

    return collage[:, :, ::-1]


def mapLabelsOneHot(data):
    data = np.asarray(data)
    class_no = int(data.max()+1)
    out = np.zeros((data.shape[0], class_no)).astype(np.float32)
    out[range(out.shape[0]), data.astype(int)] = 1
    return out


def readCIFAR(path):
    trnData = []
    trnLabels = []
    for i in range(1,6):
        with open(os.path.join(path,'data_batch_{}'.format(i)), 'rb') as f:
            try:
                data = pickle.load(f, encoding='latin1')
            except:
                data = pickle.load(f)

            trnData.append(data['data'])
            trnLabels.append(data['labels'])

    trnData = np.concatenate(trnData).reshape(-1, 3, 32, 32)
    trnData = np.concatenate([trnData[:,:,:,::-1], trnData[:,:,:,:]])
    trnLabels = np.concatenate(trnLabels)
    trnLabels = np.concatenate([trnLabels, trnLabels])

    with open(os.path.join(path,'test_batch'.format(i)), 'rb') as f:
        try:
            data = pickle.load(f, encoding='latin1')
        except:
            data = pickle.load(f)

        tstData = data['data']
        tstLabels = data['labels']

    tstData = tstData.reshape(-1, 3, 32, 32)
    tstData = np.concatenate([tstData[:,:,:,::-1], tstData[:,:,:,:]])
    tstLabels = np.concatenate([tstLabels, tstLabels])

    trnData = trnData.transpose(0, 2, 3, 1)
    tstData = tstData.transpose(0, 2, 3, 1)

    return trnData, tstData, trnLabels, tstLabels
