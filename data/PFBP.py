import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import random
import pdb
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class FBP(Dataset):
    """
    put FBP files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, setname, startidx=0):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
            mode, batchsz, n_way, k_shot, k_query, resize))

        self.path = root  # image path
        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.RandomCrop(224),
                                                 # transforms.RandomResizedCrop(224),
                                                 # transforms.RandomRotation(5),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]), ])  # resize
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]), ])

        data = []
        label = []
        user = []
        filelist = [x.strip() for x in open(setname, 'r').readlines()][1:]
        for l in filelist:
            username, filename, score = l.split(' ')
            data.append(filename)
            label.append(int(score) - 1)  # [1,2,3,4,5]--->[0,1,2,3,4]
            user.append(int(username))

        self.data = data
        self.user = np.array(user, dtype=int)
        self.label = np.array(label, dtype=int)

        self.users_unique = np.unique(self.user)  # quchong
        self.n_user = len(self.users_unique)
        self.line = [[] for i in range(self.n_user)]

        for i in range(self.n_user):
            idx = np.argwhere(self.user == self.users_unique[i]).reshape(-1)
            for j in range(self.n_way):
                idx2 = np.argwhere(self.label == j).reshape(-1)
                new_idx = np.intersect1d(idx, idx2)
                self.line[i].append(new_idx)


        self.create_batch(self.batchsz)

    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        episode here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        userlist = torch.randperm(self.n_user)

        for batchid in range(batchsz):  # for each batch
            user = userlist[batchid % self.n_user].item()

            support_x = []
            query_x = []
            for cls in range(self.n_way):
                tmp = self.line[user][cls]
                if len(tmp) >= self.k_shot + self.k_query:
                    selected_imgs_idx = np.random.choice(len(tmp), self.k_shot + self.k_query, False)
                    np.random.shuffle(selected_imgs_idx)
                    indexDtrain = np.array(selected_imgs_idx[:self.k_shot], dtype=int)  # idx for Dtrain
                    indexDtest = np.array(selected_imgs_idx[self.k_shot:], dtype=int)  # idx for Dtest
                else:
                    # print 'Warning! user:%d  class:%d  image number %d less than %d!' %(self.users_unique[user], cls, len(tmp), self.k_shot+self.k_query)
                    indexDtrain, indexDtest = self.fill_in(len(tmp))
                    # print indexDtrain, indexDtest

                support_x.append(np.array(tmp)[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(tmp)[indexDtest].tolist())

            self.support_x_batch.append(np.array(support_x).transpose())  # append set to current sets
            self.query_x_batch.append(np.array(query_x).transpose())  # append sets to current sets
        # pdb.set_trace()

    def fill_in(self, cls_num):
        # Class has fewer than k_shot + k_query images. Split the UNIQUE image
        # indices into disjoint support and query pools first (no replacement),
        # then cycle within each pool to fill the requested counts, so no image is
        # ever in both support and query. np.random.choice defaults to replacement,
        # which previously leaked an image across the split (and, in the first
        # branch, reused idx[-1] for the query that could also be in support).
        indexDtrain = np.zeros(self.k_shot, dtype=int)
        indexDtest = np.zeros(self.k_query, dtype=int)

        perm = np.random.permutation(cls_num)  # unique indices, no replacement

        if cls_num == 1:
            # Only one image exists, so it must serve both sets; unavoidable.
            indexDtrain[:] = perm[0]
            indexDtest[:] = perm[0]
            return indexDtrain, indexDtest

        # Give the support up to k_shot unique images and leave at least one for
        # the query; the two pools are disjoint slices of the same permutation.
        n_support_unique = min(self.k_shot, cls_num - 1)
        support_pool = perm[:n_support_unique]
        query_pool = perm[n_support_unique:]

        for i in range(self.k_shot):
            indexDtrain[i] = support_pool[i % len(support_pool)]
        for i in range(self.k_query):
            indexDtest[i] = query_pool[i % len(query_pool)]

        # No image may appear in both support and query (cls_num > 1).
        assert set(indexDtrain.tolist()).isdisjoint(indexDtest.tolist()), \
            "fill_in produced overlapping support/query (cls_num=%d)" % cls_num

        return indexDtrain, indexDtest

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, 224, 224)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, 224, 224)
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int)

        flatten_support_x = [os.path.join(self.path, self.data[item])
                             for sublist in self.support_x_batch[index] for item in sublist]

        flatten_query_x = [os.path.join(self.path, self.data[item])
                           for sublist in self.query_x_batch[index] for item in sublist]

        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)

        support_y = torch.arange(self.n_way).repeat(self.k_shot)
        query_y = torch.arange(self.n_way).repeat(self.k_query)

        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz


class FBP_nouser(Dataset):
    def __init__(self, root, mode, resize, setname):
        self.resize = resize  # resize to
        self.path = root  # image path
        if mode == 'train':
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.RandomCrop(224),
                                                 # transforms.RandomResizedCrop(224),
                                                 # transforms.RandomRotation(5),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]), ])  # resize
        else:
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]), ])

        data = []
        label = []
        filelist = [x.strip() for x in open(setname, 'r').readlines()][1:]
        for l in filelist:
            filename, score = l.split(' ')
            data.append(os.path.join(root, filename))
            label.append(int(score) - 1)  # [1,2,3,4,5]--->[0,1,2,3,4]

        self.data = data
        self.label = np.array(label, dtype=int)

    def __getitem__(self, index):
        img, label = self.transform(self.data[index]), torch.LongTensor([self.label[index]])
        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    pass
