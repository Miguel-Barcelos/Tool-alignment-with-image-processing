"""
Este código tem como o objetivo de realizar de criar uma máscara da imagem do gabarito realizando a leitura da quantidade de pixels dentro desta máscara, estando menor que 1200 o contorno de onde a máscara está alocada permanece verde, caso ultrapasse o valor de 1200 o contorno muda para vermelho
"""

import cv2
import numpy as np


# Inicialização da Câmera e Gabarito
camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
image_gabarito = cv2.imread(r'Gabaritos\480x480\5.png')


def pre_processamento(video):
    videopb = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(videopb, (3, 3))
    canny = cv2.Canny(blur, 50, 100)
    kernel = np.ones((3, 3), np.uint8)
    videopre = cv2.dilate(canny, kernel, iterations=2)
    videopre = cv2.erode(videopre, kernel, iterations=2)
    return videopre


while True:
    check, video = camera.read()  # Leitura do video

    if not check:  # Caso apresente erro na leitura
        break

    # Recorte para manter proporcional (480x480)
    alt_orig, lar_orig, _ = video.shape
    videoRecortado = video[0:alt_orig, 0:alt_orig]
    lar, alt, _ = videoRecortado.shape

    # Gabarito preto e branco para obter borda
    gabaritopb = cv2.cvtColor(image_gabarito, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gabaritopb', gabaritopb)
    # Obtem borda da imagem do gabarito
    bordaGabarito = cv2.Canny(image_gabarito, 1, 200)

    # Obtem os pontos onde existem bordas na imagem do gabarito
    contornoGabarito, _ = cv2.findContours(
        bordaGabarito, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Criação da máscara binária
    # Cria uma cópia de videoRecortado toda vazia
    mask = np.zeros_like(videoRecortado)
    # Faz um desenho da borda dentro da cópia vazia com preenchimento interno
    cv2.drawContours(mask, contornoGabarito, -1, (255, 255, 255), -1)
    cv2.imshow('Mask', mask)

    # Máscara aplicada ao vídeo
    # VideoRecortado preto e branco para obter borda
    videoRecortadoColor = cv2.cvtColor(videoRecortado, cv2.COLOR_BGR2RGB)
    # Interção da máscara e o video recortado
    videoMask = cv2.bitwise_and(videoRecortadoColor, mask)
    cv2.imshow('Mascara aplicada', videoMask)

    # Faz o pré processamento da máscara aplicada
    videoPreProcessado = pre_processamento(videoMask)
    cv2.imshow('videoPreProcessado1', videoPreProcessado)

    quantidadePixel = cv2.countNonZero(videoPreProcessado)
    print(quantidadePixel)

    # Lógica de ocupação
    if quantidadePixel < 9000:
        cor = (0, 0, 255)  # Vermelho
    else:
        cor = (0, 255, 0)  # Verde

    # Faz um desenho da borda na máscara aplicada
    cv2.drawContours(videoRecortado, contornoGabarito, -1, cor, 2)
    cv2.imshow('videoPreProcessado', videoRecortado)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

camera.release()
cv2.destroyAllWindows()
