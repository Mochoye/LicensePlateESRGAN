

# LicensePlateESRGAN

This is a project that uses ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) to reconstruct license plate frames from low-resolution inputs. ESRGAN is a state-of-the-art technique for image super-resolution that can produce realistic and high-quality results.

The model used for detection of License Plate can be found in [this](https://github.com/Mochoye/Licence-Plate-Detection-using-TensorFlow-Lite) repository.


## Installation

To run this project, you need to have Python 3.6 or higher and the following libraries installed:

You can install them using pip:

```bash
pip install -r requirements.txt
```



## Usage

To use this project, you need to have a video with low-resolution license plate images and a pre-trained ESRGAN model. You can download a model from [here](https://drive.google.com/file/d/1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene/view?usp=drive_link)

To run the project, use the following command:

```bash
python detector.py
```



## Results

Here are some examples of the results obtained by this project:

| Previous | After |
| ----- | ------ |
|![l1](https://github.com/Mochoye/LicensePlateESRGAN/assets/95351969/bfce2a5a-28b1-4da2-b8fe-aeced4ea71c1) | ![l2](https://github.com/Mochoye/LicensePlateESRGAN/assets/95351969/89368c28-674d-482e-b873-1649d7c1f255)
 |
| ![Input](^4^) | ![Output] |

## References

This project is based on the following paper and code:

- [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219) by Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Chen Change Loy, Yu Qiao, Xiaoou Tang
- [ESRGAN-PyTorch](https://github.com/xinntao/Real-ESRGAN) by Xinntao

## License

This project is licensed under the MIT License - see the [LICENSE] file for details.


