import nltk
import string
import os
import torch
import numpy as np
from torch.utils import data
from PIL import Image


###########################
# Dataset class definitions
###########################


class DSet(data.Dataset):
    """ Dataset Superclass """
    def __init__(
        self, split, data_df, image_dir,
        vocab, decoding_level, transform=None
    ):
        """
        Initialization

        :input:
            list_IDs:   list containing ids from entries in current split
            data_df:    pandas dataframe containing image_ids and captions,
                        indexed with ids
            image_dir:  root file for images
            vocab:      file containing id assignment for vocabulary entries
            transform:  torchvision image transformation wrapper
        """

        self.df = data_df
        self.split = split
        self.split_df = self.df.loc[self.df.split == self.split]
        self.list_IDs = self.split_df.index.to_list()

        self.image_dir = image_dir
        self.vocab = vocab

        self.transform = transform
        self.decoding_level = decoding_level

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_IDs)


class COCOCaptionDataset(DSet):
    """
        DSet subclass for COCO captions
        Inherits: __init__ and __len__ methods
        Provides: __getitem__ method for full images + COCO captions.
    """

    def __getitem__(self, index):
        """
        Generates one data pair
        :input: index
        :output: image, target caption
        """
        # select sample
        ID = self.list_IDs[index]
        entry = self.df.loc[ID]

        # get image filename
        img_path = os.path.join(
            self.image_dir, '{coco_split}2014/',
            filename_from_id(entry.image_id, prefix='COCO_{coco_split}2014_')
            )
        img_path = img_path.format(coco_split=entry.coco_split)
        # read image file and transform
        image = Image.open(img_path).convert('RGB')
        # transform image
        if self.transform is not None:
            image = self.transform(image)

        if self.decoding_level == 'word':
            # convert caption to word ids
            target = caption_to_word_id(entry.caption, self.vocab)
        elif self.decoding_level == 'char':
            # convert caption to character ids
            target = caption_to_char_id(entry.caption, self.vocab)
        else:
            raise Exception("Invalid decoding level " + self.decoding_level)

        target_len = len(target)

        return image, target, target_len


class RefCOCODataset(DSet):
    """
        DSet subclass for referring expressions from RefCOCO / RefCOCO+
        Inherits: __init__ and __len__ methods
        Provides: __getitem__ method for full images + RefCOCO/RefCOCO+ Refs.
    """

    def __getitem__(self, index):
        """
        Generates one data pair
        :input: index
        :output: image, target caption, position vector
        """
        # select sample
        ID = self.list_IDs[index]
        entry = self.df.loc[ID]

        # get image filename
        img_path = os.path.join(
            self.image_dir, '{coco_split}2014/',
            filename_from_id(entry.image_id, prefix='COCO_{coco_split}2014_')
            )
        img_path = img_path.format(coco_split=entry.coco_split)
        # read image file and transform
        image = Image.open(img_path).convert('RGB')
        # compute position vector
        pos_features = compute_position_features(image, entry.bbox)
        # crop image to bounding box
        image = crop_image_to_bb(image, entry.bbox)
        # transform image
        if self.transform is not None:
            image = self.transform(image)

        if self.decoding_level == 'word':
            # convert caption to word ids
            target = caption_to_word_id(entry.caption, self.vocab)
        elif self.decoding_level == 'char':
            # convert caption to character ids
            target = caption_to_char_id(entry.caption, self.vocab)
        else:
            raise Exception("Invalid decoding level " + self.decoding_level)

        target_len = len(target)

        return image, target, pos_features, target_len


##############################
# Collate function definitions
##############################


def captions_collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (image, caption).

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, x_dim, y_dim).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, x_dim, y_dim).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # sort data by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, _ = zip(*data)

    # merge images (from tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # merge captions (from tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    # create empty tensor and fill in caption ints
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths


def reg_collate_fn(data):
    """
    see captions_collate_fn, can handle position information
    """
    # same stuff as in captions_collate_fn
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, pos_features, _ = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    # collate position information data
    positions = torch.stack(pos_features, 0)

    return images, targets, positions, lengths


#######################
# loader initialization
#######################


def get_caption_loader(
        loader_type, decoding_level, list_IDs,
        data_df, image_dir, vocab,
        transform, batch_size, shuffle,
        num_workers, drop_last=False
        ):
    """Returns torch.utils.data.DataLoader for dataset."""

    # initialize dataset

    dataset = COCOCaptionDataset(
        list_IDs=list_IDs, data_df=data_df, image_dir=image_dir,
        vocab=vocab, transform=transform, decoding_level=decoding_level
        )

    # initialize dataloader for dataset
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=captions_collate_fn
        )

    return data_loader


def get_reg_loader(
        loader_type, decoding_level, list_IDs,
        data_df, image_dir, vocab,
        transform, batch_size, shuffle,
        num_workers, drop_last=False
        ):
    """Returns torch.utils.data.DataLoader for dataset."""

    # initialize dataset

    dataset = RefCOCODataset(
        list_IDs=list_IDs, data_df=data_df, image_dir=image_dir,
        vocab=vocab, transform=transform, decoding_level=decoding_level
        )

    # initialize dataloader for dataset
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=reg_collate_fn
        )

    return data_loader


##################
# Cluster Loader #
##################


class COCOClusters():
    """
    returns data for image clusters

    __getitem__ method:
        returns caption_ids, image_ids, and image data
        for cluster entries given a cluster id
    """
    def __init__(self, image_clusters,
                 decoding_level, list_IDs,
                 data_df, image_dir, vocab, transform):

        # initialize dataset wrapper
        self.dataset = COCOCaptionDataset(
            decoding_level=decoding_level, list_IDs=list_IDs, data_df=data_df,
            image_dir=image_dir, vocab=vocab, transform=transform
        )
        self.image_clusters = image_clusters

    def __len__(self):
        return len(self.image_clusters)

    def __getitem__(self, index):
        """
            ::input:: index for cluster in self.image_clusters
            ::output:: caption_ids, image_ids and image data for that cluster
        """
        # retrieve indices in id list, caption ids
        # and image ids for entries in cluster
        id_list_idx, caption_ids, image_ids = self.cluster_ids(index)
        # retrieve image data from self.dataset for the given indices
        images = [self.dataset[i][0] for i in id_list_idx]

        return list(zip(caption_ids, image_ids, images))

    def cluster_ids(self, i):
        df = self.dataset.df
        # select cluster from image_clusters
        image_ids = self.image_clusters[i]

        # retrieve caption_ids from data frame, given the image ids:
        # 1) set caption_id as column (not index)
        # 2) select first entry for each image id
        # 3) restore order from image_ids list
        # 4) get caption ids as list
        caption_ids = df.loc[df.image_id.isin(image_ids)]\
            .reset_index()\
            .groupby('image_id').agg('first')\
            .loc[image_ids]\
            .id.to_list()

        # get position from caption_ids in dataset id list
        # (used for retrieving entries from dataset wrapper)
        ids_idx = [
            self.dataset.list_IDs.index(i)
            for i in caption_ids
        ]

        return(ids_idx, caption_ids, image_ids)


class RefCOCOClusters():
    """
    returns data for targets and distractors in refcoco(+) images

    __getitem__ method:
        returns caption_ids, image_ids, and image data
        for target and distractors given a target index
    image_entities method:
        returns caption_ids, image_ids, and image data
        for objects in an image given an image id
    """

    def __init__(self, decoding_level, list_IDs,
                 data_df, image_dir, vocab, transform):

        # initialize dataset wrapper
        self.dataset = RefCOCODataset(
            decoding_level=decoding_level, list_IDs=list_IDs, data_df=data_df,
            image_dir=image_dir, vocab=vocab, transform=transform
        )

    def __getitem__(self, index):
        """
            ::input:: index for cluster in self.image_clusters
            ::output:: sent_ids, image_ids and image data for that cluster
        """
        # retrieve indices in id list, caption ids and image ids for entries in cluster
        id_list_idx,sent_ids = self.get_distractors(index)
        # retrieve image data from self.dataset for the given indices
        image_data = [self.dataset[i] for i in id_list_idx]
        # unpack image data
        images = [entry[0] for entry in image_data]
        pos_infs = [entry[2] for entry in image_data]

        return list(zip(sent_ids, images, pos_infs))

    def image_entities(self, img_id):
        """
            ::input:: image_id
            ::output:: sent_ids and image data for objects in image
        """
        df = self.dataset.df
        df = df.reset_index()
        # get entries in image
        image_entries = df.loc[df.image_id == img_id]
        # get first sent_id for every object in image
        sent_ids = image_entries.reset_index()\
            .groupby('ann_id')\
            .agg('first')\
            .sent_id.to_list()

        # get position from sent_ids in dataset id list
        # (used for retrieving entries from dataset wrapper)
        ids_idx = [
            i for i, c_id in enumerate(self.dataset.list_IDs)
            if c_id in sent_ids
            ]

        # retrieve image data from self.dataset for the given indices
        image_data = [self.dataset[i] for i in ids_idx]
        # unpack image data
        images = [entry[0] for entry in image_data]
        pos_infs = [entry[2] for entry in image_data]

        return list(zip(sent_ids, images, pos_infs))

    def get_distractors(self, i):
        """
            ::input:: id for target object
            ::output:: sent_ids and positions in dataset id list
                       for target and distractors in same image
        """

        df = self.dataset.df
        # get target entry
        target = df.loc[self.dataset.list_IDs[i]]
        target_id = target.name

        # get image_id from target entry
        image_id = target.image_id
        # get entries in same image
        image_entries = df.loc[df.image_id == image_id]

        # get first sent_id for every distractor object in image

        distractor_ids = image_entries.reset_index()\
            .groupby('ann_id')\
            .agg('first').drop(target.ann_id)\
            .sent_id.to_list()

        # combine target and distractor ids
        sent_ids = [target_id] + distractor_ids

        # get position from sent_ids in dataset id list
        # (used for retrieving entries from dataset wrapper)
        ids_idx = [
            i for i, c_id in enumerate(self.dataset.list_IDs)
            if c_id in sent_ids
            ]

        return(ids_idx, sent_ids)


################
# util functions
################


def iterate_targets(lst):
    """
    set every entry in lst as target once
    use other entries as distractors
    """
    for i in range(len(lst)):
        # get target + distractors at current step
        target = lst[i]
        distractors = lst[:i] + lst[i+1:]

        # return items in current order
        yield [target] + distractors


def caption_to_word_id(caption, vocab):
    """ create target array with ids from vocabulary for words in caption """

    # remove punctuation
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    # whitespace
    caption = ' '.join(caption.split())

    tokens = nltk.tokenize.word_tokenize(str(caption).lower())
    target = []
    target.append(vocab('<start>'))
    target.extend([vocab(token) for token in tokens])
    target.append(vocab('<end>'))
    target = torch.Tensor(target)

    return target


def caption_to_char_id(caption, vocab):
    """ create target array with ids from vocabulary for chars in caption """

    # remove punctuation
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    # whitespace
    caption = ' '.join(caption.split())

    tokens = [char for char in str(caption).lower()]
    target = []
    target.append(vocab('^')) # start token
    target.extend([vocab(token) for token in tokens])
    target.append(vocab('$')) # stop token
    target = torch.Tensor(target)

    return target


def filename_from_id(image_id, prefix='', file_ending='.jpg'):
    """
    get image filename from id: pad image ids with zeroes,
    add file prefix and file ending
    """
    padded_ids = str(image_id).rjust(12, '0')
    filename = prefix + padded_ids + file_ending

    return (filename)


def crop_image_to_bb(image, bb):
    """
    crop image to bounding box annotated for the current region
    :input:
        Image (PIL Image)
        Bounding Box coordinates (list containing values for x, y, w, h)
    :output:
        Image (PIL Image) cropped to bounding box coordinates
    """

    # convert image into numpy array
    image_array = np.array(image)

    # get bounding box coordinates (round since integers are needed)
    x, y, w, h = round(bb[0]), round(bb[1]), round(bb[2]), round(bb[3])

    # calculate minimum and maximum values for x and y dimension
    x_min, x_max = x, x+w
    y_min, y_max = y, y+h

    # crop image by slicing image array
    image_cropped = image_array[y_min:y_max, x_min:x_max, :]

    return (Image.fromarray(image_cropped))


def compute_position_features(image, bb):
    """
    compute position features of bounding box within image
    7 features (all relative to image dimensions):
        - x and y coordinates of bb corner points ((x1,y1) and (x2,y2))
        - bb area
        - distance between bb center and image center
        - image orientation
    :input:
        Image (PIL Image)
        Bounding Box coordinates (list containing values for x, y, w, h)
    :output:
        numpy array containing the features computed

    https://github.com/clp-research/clp-vision/blob/master/ExtractFeats/extract.py
    """

    image = np.array(image)
    # get image dimensions, split up list containing bb values
    ih, iw, _ = image.shape
    x, y, w, h = bb

    # x and y coordinates for bb corners
    # upper left
    x1r = x / iw
    y1r = y / ih
    # lower right
    x2r = (x+w) / iw
    y2r = (y+h) / ih

    # bb area
    area = (w*h) / (iw*ih)

    # image orientation / ratio between image height and width
    ratio = iw / ih

    # distance between bb center and image center
    cx = iw / 2
    cy = ih / 2
    bcx = x + w / 2
    bcy = y + h / 2
    distance = np.sqrt((bcx-cx)**2 + (bcy-cy)**2) / np.sqrt(cx**2+cy**2)

    return torch.Tensor([x1r, y1r, x2r, y2r, area, ratio, distance])
