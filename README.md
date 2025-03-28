# Sign Language Datasets Collection

This repository provides easy access to various sign language datasets through Hugging Face datasets. Our goal is to make sign language datasets more accessible to researchers, developers, and the community.

## Available Datasets

| Dataset Name | Language | Description | Size | Reference |
|--------------|----------|-------------|------|-----------|
| How2Sign | American Sign Language (ASL) | Large-scale multimodal dataset containing American Sign Language videos aligned with English speech and text. Includes instructional videos covering various topics. | 35,000 videos | [How2Sign Paper](https://arxiv.org/abs/2008.08143) |
| RWTH-PHOENIX-Weather 2014T | German Sign Language (DGS) | Contains weather forecast recordings from German public TV performed by professional sign language interpreters. Features continuous sign language, gloss annotations, and translations. | 8,257 videos | [PHOENIX Paper](https://www-i6.informatik.rwth-aachen.de/publications/download/1064/Koller-LREC-2016.pdf) |

## Usage

Each dataset is available through the Hugging Face datasets library. Example usage:

```python
from datasets import load_dataset

# Load How2Sign dataset
how2sign = load_dataset("how2sign")

# Load PHOENIX dataset
phoenix = load_dataset("rwth-phoenix-weather-2014t")
```

## Contributing

We welcome contributions! If you have a dataset to add or improvements to suggest, please open an issue or submit a pull request.

## License

Please note that each dataset has its own license. Make sure to check and comply with the respective dataset licenses before use.

## Citation

When using these datasets, please cite the original papers:

### How2Sign
```bibtex
@inproceedings{duarte2021how2sign,
    title={How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language},
    author={Duarte, Amanda and Palaskar, Shruti and Ventura, Lucas and Ghadiyaram, Deep and DeHaan, Kenneth and Metze, Florian and Torres, Jorge and Giró-i-Nieto, Xavier},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}
}
```

### RWTH-PHOENIX-Weather 2014T
```bibtex
@inproceedings{koller2015continuous,
    title={Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers},
    author={Koller, Oscar and Forster, Jens and Ney, Hermann},
    booktitle={Computer Vision and Image Understanding},
    year={2015}
}