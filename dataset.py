
import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import tarfile
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
from scipy import io as mat_io
from skimage import io
from torchvision.transforms import transforms
import scipy.io as sio
from subprocess import call
import h5py
from read_annotations import GetAnnotBoxLoc


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class CUBDataset(MNIST):
    #urls = ['http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz']
    urls = ['file:///home/song/song/CUB_200_2011.tgz']
    train_transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.repeat(3,1,1)),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.RandomSizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.repeat(3,1,1)),
        normalize
    ])
    
    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.raw_folder))
            os.makedirs(os.path.join(self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        folder = os.path.join(self.raw_folder)

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())

            tar = tarfile.open(file_path, "r:gz")
            tar.extractall(folder)
            tar.close()

        folder += '/CUB_200_2011'

        # process and save as torch files
        print('Processing...')

        training_set = []
        testing_set = []

        with open(folder+'/train_test_split.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                image_id, cat = line.split(' ')
                cat = cat[0]
                if cat == '1':
                    training_set.append(image_id)
                else:
                    testing_set.append(image_id)

        class_dict = {}
        with open(folder+'/classes.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id, class_name = line.split(' ')
                class_dict[class_id] = class_name[:-1]

        images_dict = {}
        with open(folder+'/images.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_id, img_name = line.split(' ')

                images_dict[img_id] = img_name[:-1]

        image_class_dict = {}
        with open(folder+'/image_class_labels.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_id, class_id = line.split(' ')
                image_class_dict[img_id] = class_id[:-1]

        bbox_dict = {}
        with open(folder+'/bounding_boxes.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_id, x, y, w, h = line.split(' ')
                bbox_dict[img_id] = (float(x), float(y), float(w), float(h[:-1]))

        train_data, train_label = [], []
        test_data, test_label = [], []

        for i, img_id in enumerate(training_set):
            # print(i, img_id)
            x = np.array(Image.open(folder + '/images/' + images_dict[img_id]))
            bbox = tuple(int(x) for x in bbox_dict[img_id])
            if x.shape[-1] == 1:
                x = x.repeat(3, -1)
            x = x[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            train_data.append(torch.from_numpy(x))
            train_label.append(torch.LongTensor([int(image_class_dict[img_id])]))

        for i, img_id in enumerate(testing_set):
            # print(i, img_id)
            x = np.array(Image.open(folder + '/images/' + images_dict[img_id]))
            bbox = tuple(int(x) for x in bbox_dict[img_id])
            x = x[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            test_data.append(torch.from_numpy(x))
            test_label.append(torch.LongTensor([int(image_class_dict[img_id])]))

        training_set = (train_data, train_label)
        test_set = (test_data, test_label)

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if len(img.shape) != 3:
            img = Image.fromarray(img.numpy())
            img = img.convert('RGB')
        else:
            img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = target - 1
        #if img.size(0) == 1:
            #img = img.expand(3, 224, 224)

        return img, target


class VOCPartDataset(MNIST):
    urls = ['http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar']    #'http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz',
    train_transform = test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.raw_folder))
            os.makedirs(os.path.join(self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        folder = os.path.join(self.raw_folder)

        '''for url in self.urls:
            print('Downloading ' + url)
            #data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            #with open(file_path, 'wb') as f:
                #f.write(data.read())
       
            if file_path.endswith('tar.gz'):
                tar = tarfile.open(file_path, "r:gz")
            else:
                tar = tarfile.open(file_path, 'r:')
            tar.extractall(folder)
            tar.close()'''

        # process and save as torch files
        print('Processing...')

        '''image_folder = folder + '/VOCdevkit/VOC2010/JPEGImages'
        annotation_folder = folder + '/Annotations_Part'

        training_set = []
        testing_set = []

        for f in os.listdir(annotation_folder):
            file_name = annotation_folder+'/'+f
            file = sio.loadmat(file_name, squeeze_me=True)

            pass

        with open(folder+'/train_test_split.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                image_id, cat = line.split(' ')
                cat = cat[0]
                if cat == '1':
                    training_set.append(image_id)
                else:
                    testing_set.append(image_id)

        class_dict = {}
        with open(folder+'/classes.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id, class_name = line.split(' ')
                class_dict[class_id] = class_name[:-1]

        images_dict = {}
        with open(folder+'/images.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_id, img_name = line.split(' ')

                images_dict[img_id] = img_name[:-1]

        image_class_dict = {}
        with open(folder+'/image_class_labels.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_id, class_id = line.split(' ')
                image_class_dict[img_id] = class_id[:-1]

        bbox_dict = {}
        with open(folder+'/bounding_boxes.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_id, x, y, w, h = line.split(' ')
                bbox_dict[img_id] = (float(x), float(y), float(w), float(h[:-1]))'''

        train_data, train_label = [], []
        test_data, test_label = [], []
        image_name_train = np.load('image_name.npy')              #from part_VOC.py
        label_train = np.load('label.npy')
        image_name_test = np.load('image_name_test.npy')
        label_test = np.load('label_test.npy')
        classes = ['bird','cat','cow','dog','horse','sheep']
        for i in range(len(image_name_train)):
            # print(i, img_id)
            x = np.array(Image.open(image_name_train[i]))
            image_name = image_name_train[i].split('/')[-1].split('.')[0] + '.xml'
            box_address = './data/VOCdevkit/VOC2012/Annotations/' + image_name
            ObjBndBoxSet=GetAnnotBoxLoc(box_address)
            for key in ObjBndBoxSet.keys():
                if key == classes[int(label_train[i]-1)]:
                    bbox = ObjBndBoxSet[key][0]
                #bbox = tuple(int(x) for x in bbox_dict[img_id])
                if x.shape[-1] == 1:
                    x = x.repeat(3, -1)
                x = x[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                train_data.append(torch.from_numpy(x))
                train_label.append(torch.LongTensor([int(label_train[i])]))
                break

        for i in range(len(image_name_test)):
            # print(i, img_id)
            x = np.array(Image.open(image_name_test[i]))
            image_name = image_name_test[i].split('/')[-1].split('.')[0] + '.xml'
            box_address = './data/VOCdevkit/VOC2012/Annotations/' + image_name
            ObjBndBoxSet=GetAnnotBoxLoc(box_address)
            for key in ObjBndBoxSet.keys():
                if key == classes[int(label_test[i]-1)]:
                    bbox = ObjBndBoxSet[key][0]
            #bbox = tuple(int(x) for x in bbox_dict[img_id])
                x = x[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                test_data.append(torch.from_numpy(x))
                test_label.append(torch.LongTensor([int(label_test[i])]))
                break

        training_set = (train_data, train_label)
        test_set = (test_data, test_label)

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = target - 1
        if img.size(0) == 1:
            img = img.expand(3, 224, 224)

        return img, target


class ImagenetPartDataset(MNIST):
    urls = ['https://github.com/zqs1022/detanimalpart.git']
    train_transform = test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        folder = os.path.join(self.root, self.raw_folder)

        # for url in self.urls:
        #     print('Downloading ' + url)
        #     current = os.cwd()
        #     os.chdir(folder)
        #     call(['git', 'clone', url])
        #     os.chdir(current)

        # process and save as torch files
        print('Processing...')

        training_set = []
        testing_set = []

        for f in os.listdir(folder):
            img_folder = folder + '/' + f + '/img/img'
            data_file = folder + '/' + f + '/img/data.mat'

            with h5py.File(data_file, 'r') as f:
                files = list(f)
                pass

        with open(folder+'/train_test_split.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                image_id, cat = line.split(' ')
                cat = cat[0]
                if cat == '1':
                    training_set.append(image_id)
                else:
                    testing_set.append(image_id)

        class_dict = {}
        with open(folder+'/classes.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id, class_name = line.split(' ')
                class_dict[class_id] = class_name[:-1]

        images_dict = {}
        with open(folder+'/images.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_id, img_name = line.split(' ')

                images_dict[img_id] = img_name[:-1]

        image_class_dict = {}
        with open(folder+'/image_class_labels.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_id, class_id = line.split(' ')
                image_class_dict[img_id] = class_id[:-1]

        bbox_dict = {}
        with open(folder+'/bounding_boxes.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_id, x, y, w, h = line.split(' ')
                bbox_dict[img_id] = (float(x), float(y), float(w), float(h[:-1]))

        train_data, train_label = [], []
        test_data, test_label = [], []

        for i, img_id in enumerate(training_set):
            # print(i, img_id)
            x = np.array(Image.open(folder + '/images/' + images_dict[img_id]))
            bbox = tuple(int(x) for x in bbox_dict[img_id])
            if x.shape[-1] == 1:
                x = x.repeat(3, -1)
            x = x[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            train_data.append(torch.from_numpy(x))
            train_label.append(torch.LongTensor([int(image_class_dict[img_id])]))

        for i, img_id in enumerate(testing_set):
            # print(i, img_id)
            x = np.array(Image.open(folder + '/images/' + images_dict[img_id]))
            bbox = tuple(int(x) for x in bbox_dict[img_id])
            x = x[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            test_data.append(torch.from_numpy(x))
            test_label.append(torch.LongTensor([int(image_class_dict[img_id])]))

        training_set = (train_data, train_label)
        test_set = (test_data, test_label)

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = target - 1
        if img.size(0) == 1:
            img = img.expand(3, 224, 224)

        return img, target

class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, mode, data_dir, metas):

        self.data_dir = data_dir
        self.data = []
        self.target = []

        self.to_tensor = transforms.ToTensor()
        self.mode = mode

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):

            # self.data.append(img_resized)
            self.data.append(data_dir + img_[5][0])
            # if self.mode == 'train':
            self.target.append(img_[4][0][0])

        if self.mode == "train":
            print("Load-up dataset with Auto Augment")
            self.train_transform = transforms.Compose([
                    #transforms.Resize((224, 224)),
                    transforms.RandomSizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Lambda(lambda x: x.repeat(3,1,1)),
                    normalize
                ])
        self.val_or_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.RandomSizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x.repeat(3,1,1)),
            normalize
        ])

    '''def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.mode == 'train':
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = target - 1
        if img.size(0) == 1:
            img = img.expand(3, 224, 224)

        return img, target'''


    def __getitem__(self, idx):
        img = io.imread(self.data[idx])
        if len(img.shape) != 3:
            image = Image.fromarray(img)
            image = image.convert('RGB')
        else:
            image = Image.fromarray(img)
        #target = target - 1
        if self.mode == 'train':
            image = self.train_transform(image)
        elif self.mode == 'test':
                image = self.val_or_test_transform(image)
            
        return image,torch.tensor(self.target[idx]-1, dtype=torch.long)

    def __len__(self):
        return len(self.data)

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID,txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.img_list[idx][1])}
        return sample

if __name__ == '__main__':
    dataset = ImagenetPartDataset('data/ImagenetPart', download=True)
    for example in dataset:
        pass
