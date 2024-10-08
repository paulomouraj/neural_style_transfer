# Neural Style Transfer of Artworks

## Description

The goal of this work was to transfer the style of a given work-of-art into an arbitrary image using a pre-trained convolutional neural network (CNN), here chosen as VGG-19. Moreover, an extra feature was implemented to keep the colors of the arbitrary image or transfer only the colors of a third image. This project was part of [Télécom Paris Image Processing](https://www.telecom-paris.fr/fr/ingenieur/formation/2e-annee-orientation/image) course.  

Examples of the implemented method, where the target column is the output:  
![Set of examples](https://github.com/paulomouraj/neural_style_transfer/blob/main/examples/set_of_examples1.png)

## Dataset

The artwork images were taken from the dataset available on: https://huggingface.co/datasets/huggan/wikiart?row=0

## Requirements

- Python 3.10
- NumPy 1.26.0
- PyTorch 2.3.0
- PyTorch-cuda 11.8
- Pillow 10.4.0
- TorchVision 0.18.0
- OpenCV 4.8.1

## Contributors

Paulo Roberto de Moura Júnior (me)  
Giovanni Benedetti da Rosa  
Cristian Alejandro Chávez Becerra  
Juan Esteban Rios Gallego

[High level description](https://github.com/paulomouraj/neural_style_transfer/blob/main/docs/neural_style_transfer_HLD.pdf)
