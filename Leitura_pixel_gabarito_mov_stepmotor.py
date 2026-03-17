import cv2
import numpy as np
import nidaqmx
import time
import threading
from nidaqmx.constants import LineGrouping

# ==============================================================================
# 1. CONFIGURAÇÕES GERAIS E DAQ
# ==============================================================================
# Ajuste o caminho da imagem do gabarito aqui:
CAMINHO_GABARITO = r'Gabaritos\480x480\5.png'
LIMIAR_PIXELS = 4000  # Valor de corte entre Livre e Ocupado

# Configurações da NI DAQ
NOME_DO_DISPOSITIVO = "Dev1"
DIR_PIN = f"{NOME_DO_DISPOSITIVO}/port0/line0"
STEP_PIN = f"{NOME_DO_DISPOSITIVO}/port0/line1"
ENABLE_PIN = f"{NOME_DO_DISPOSITIVO}/port0/line2"

NUM_PASSOS = 800
TEMPO_PULSO = 2000.0 / 1000000.0  # 0.002 segundos
DELAY_SEGUNDOS = 0.5

# Variável de estado para evitar disparos repetidos
# Estados possíveis: "INDEFINIDO", "LIVRE", "OCUPADO"
estado_anterior = "INDEFINIDO"

# Lock para thread do motor
motor_lock = threading.Lock()

# ==============================================================================
# 2. FUNÇÕES DE VISÃO (Processamento de Imagem)
# ==============================================================================


def pre_processamento_mask(video):
    """Aplica filtros na imagem já mascarada para contagem de pixels."""
    videopb = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(videopb, (3, 3))
    canny = cv2.Canny(blur, 50, 100)
    kernel = np.ones((3, 3), np.uint8)
    videopre = cv2.dilate(canny, kernel, iterations=2)
    videopre = cv2.erode(videopre, kernel, iterations=2)
    return videopre


def criar_mascara_gabarito(caminho_imagem, shape_ref):
    """
    Gera a máscara binária baseada na imagem do gabarito.
    Executada apenas uma vez no início para otimizar.
    """
    try:
        img_gabarito = cv2.imread(caminho_imagem)
        if img_gabarito is None:
            raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")

        # Processa o gabarito para achar o contorno
        gabaritopb = cv2.cvtColor(img_gabarito, cv2.COLOR_BGR2GRAY)
        bordaGabarito = cv2.Canny(gabaritopb, 1, 200)
        contornoGabarito, _ = cv2.findContours(
            bordaGabarito, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Cria máscara vazia do tamanho do vídeo recortado
        mask = np.zeros(shape_ref[:2], dtype=np.uint8)

        # Desenha o contorno preenchido na máscara (branco = área de interesse)
        cv2.drawContours(mask, contornoGabarito, -1, (255), -1)

        return mask, contornoGabarito
    except Exception as e:
        print(f"Erro ao criar máscara: {e}")
        return None, None

# ==============================================================================
# 3. FUNÇÕES DO MOTOR (Thread)
# ==============================================================================


def mover_motor(direcao_high):
    with motor_lock:
        try:
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(
                    DIR_PIN, line_grouping=LineGrouping.CHAN_PER_LINE)
                task.do_channels.add_do_chan(
                    STEP_PIN, line_grouping=LineGrouping.CHAN_PER_LINE)
                task.do_channels.add_do_chan(
                    ENABLE_PIN, line_grouping=LineGrouping.CHAN_PER_LINE)

                # Habilitar e configurar direção
                task.write([direcao_high, False, False], auto_start=True)
                time.sleep(0.05)

                # Loop de Pulsos
                for _ in range(NUM_PASSOS):
                    task.write([direcao_high, True, False])
                    time.sleep(TEMPO_PULSO)
                    task.write([direcao_high, False, False])
                    time.sleep(TEMPO_PULSO)

                time.sleep(DELAY_SEGUNDOS)

                # Desabilitar motor
                task.write([False, False, True])
                print(f"[{threading.current_thread().name}] Movimento finalizado.")

        except nidaqmx.errors.DaqError as e:
            print(f"ERRO DAQ: {e}")


def girar_para_ocupado():
    """Gira sentido Horário (Exemplo) quando detecta ocupação"""
    print(">>> AÇÃO: Girando para posição OCUPADO")
    mover_motor(direcao_high=True)


def girar_para_livre():
    """Gira sentido Anti-Horário (Exemplo) quando libera"""
    print(">>> AÇÃO: Girando para posição LIVRE")
    mover_motor(direcao_high=False)

# ==============================================================================
# 4. LOOP PRINCIPAL
# ==============================================================================


def main():
    global estado_anterior

    # Inicializa Câmera (Tente 0 ou 1 dependendo da sua porta USB)
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    # Leitura inicial para pegar dimensões
    if not camera.isOpened():
        print("Erro ao abrir câmera.")
        return

    _, frame_inicial = camera.read()
    # Recorte para manter proporcional (conforme seu código original 480x480)
    alt_orig, _, _ = frame_inicial.shape
    # Assume quadrado baseado na altura
    recorte_shape = (alt_orig, alt_orig, 3)

    # --- PREPARAÇÃO DA MÁSCARA (Executada 1 vez) ---
    print("Carregando gabarito e gerando máscara...")
    mask_binaria, contornos_desenho = criar_mascara_gabarito(
        CAMINHO_GABARITO, recorte_shape)

    if mask_binaria is None:
        return

    try:
        while True:
            check, video = camera.read()
            if not check:
                break

            # 1. Recorte do Vídeo (0 a Altura, 0 a Altura) -> Quadrado
            videoRecortado = video[0:alt_orig, 0:alt_orig]

            # 2. Aplicação da Máscara
            # A máscara precisa ser aplicada bit a bit
            video_masked = cv2.bitwise_and(
                videoRecortado, videoRecortado, mask=mask_binaria)

            # 3. Pré-processamento e Contagem
            imagem_processada = pre_processamento_mask(video_masked)
            quantidadePixel = cv2.countNonZero(imagem_processada)

            # 4. Lógica de Estado (LIVRE ou OCUPADO)
            # Baseado no seu script: < 6350 era Vermelho (Ocupado/Diferente), >= era Verde (Livre/Igual)
            if quantidadePixel < LIMIAR_PIXELS:
                estado_atual = "OCUPADO"
                cor_borda = (0, 0, 255)  # Vermelho
            else:
                estado_atual = "LIVRE"
                cor_borda = (0, 255, 0)  # Verde

            # 5. Lógica de Decisão e Controle do Motor
            # Só age se o estado mudou E se o motor não está rodando agora
            motor_ativo = any(t.name == 'MotorThread' and t.is_alive()
                              for t in threading.enumerate())

            if estado_atual != estado_anterior:
                if not motor_ativo:
                    if estado_atual == "OCUPADO":
                        # Mudou de Livre para Ocupado
                        t = threading.Thread(
                            target=girar_para_ocupado, name='MotorThread')
                        t.start()
                    elif estado_atual == "LIVRE":
                        # Mudou de Ocupado para Livre
                        t = threading.Thread(
                            target=girar_para_livre, name='MotorThread')
                        t.start()

                    # Atualiza o estado apenas se disparou (ou se inicialização)
                    estado_anterior = estado_atual

            # --- VISUALIZAÇÃO ---
            # Desenha o contorno do gabarito na imagem original recortada para feedback visual
            if contornos_desenho:
                cv2.drawContours(
                    videoRecortado, contornos_desenho, -1, cor_borda, 2)

            # Textos informativos
            cv2.putText(videoRecortado, f"Pixels: {quantidadePixel}", (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, cor_borda, 2)
            cv2.putText(videoRecortado, f"Estado: {estado_atual}", (10, 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, cor_borda, 2)

            cv2.imshow('Sistema de Visao + Motor', videoRecortado)
            # cv2.imshow('Processado (Debug)', imagem_processada) # Descomente para ver o que o PC vê

            if cv2.waitKey(1) & 0xFF == ord('e'):
                break

    except Exception as e:
        print(f"Erro Geral: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        # Garantir motor desabilitado ao fechar
        try:
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(DIR_PIN)
                task.do_channels.add_do_chan(STEP_PIN)
                task.do_channels.add_do_chan(ENABLE_PIN)
                task.write([False, False, True], auto_start=True)
        except:
            pass


if __name__ == "__main__":
    main()
