
import torch




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, class_mapping):
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.class_mapping = class_mapping
        self.reverse_class_mapping = {v:k for (k,v) in self.class_mapping.items()}
        self.image_shape = (3,224,224)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        feature = self.feature_extractor(example['image'], return_tensors="pt")
        label = torch.tensor(self.reverse_class_mapping[example['dx']])
        return feature['pixel_values'].view(self.image_shape), label


