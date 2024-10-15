import cv2
import numpy as np
from sklearn.neighbors import KDTree

# Dicionário de cores do cubo mágico em RGB
color_dict = {
    "Vermelho": (255, 0, 0),
    "Verde": (0, 255, 0),
    "Azul": (0, 0, 255),
    "Amarelo": (255, 255, 0),
    "Laranja": (255, 165, 0),
    "Branco": (255, 255, 255)
}

# Converter RGB para o nome da cor
def convert_rgb_to_names(rgb_tuple):
    names = list(color_dict.keys())
    rgb_values = list(color_dict.values())
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query([rgb_tuple])
    
    return names[index[0][0]]

# Função para desenhar o grid de quadrados fechados no centro da imagem
def desenhar_grid_quadrado(img, linhas=3, colunas=3, proporcao=0.6):
    height, width, _ = img.shape
    lado_grid = int(min(height, width) * proporcao)
    
    start_y = (height - lado_grid) // 2
    start_x = (width - lado_grid) // 2
    
    tamanho_linha = lado_grid // linhas
    tamanho_coluna = lado_grid // colunas

    # Desenhar o grid
    for i in range(linhas):
        for j in range(colunas):
            x_start = start_x + j * tamanho_coluna
            y_start = start_y + i * tamanho_linha
            x_end = x_start + tamanho_coluna
            y_end = y_start + tamanho_linha
            cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    return img, start_x, start_y, tamanho_linha, tamanho_coluna

# Inicializa a webcam
cap = cv2.VideoCapture(0)

# Definir o número de linhas e colunas (3x3)
linhas = 3
colunas = 3
proporcao = 0.6  # O grid ocupará 60% da área central
tamanho_reconhecimento = 0.2  # Proporção da área de reconhecimento dentro de cada quadrado

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Desenhar o grid de quadrados no centro do frame
    frame_com_grid, start_x, start_y, tamanho_linha, tamanho_coluna = desenhar_grid_quadrado(frame.copy(), linhas, colunas, proporcao)

    # Converter para o espaço RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Matriz para armazenar as cores
    matriz_cores = []

    for i in range(linhas):
        linha_cores = []
        for j in range(colunas):
            # Definir as coordenadas da seção de reconhecimento
            y_start = start_y + i * tamanho_linha + int(tamanho_linha * (1 - tamanho_reconhecimento) / 2)
            y_end = start_y + i * tamanho_linha + int(tamanho_linha * (1 + tamanho_reconhecimento) / 2)
            x_start = start_x + j * tamanho_coluna + int(tamanho_coluna * (1 - tamanho_reconhecimento) / 2)
            x_end = start_x + j * tamanho_coluna + int(tamanho_coluna * (1 + tamanho_reconhecimento) / 2)
            
            # Extrair a seção da imagem para reconhecimento
            hsv_section = frame[y_start:y_end, x_start:x_end]
            rgb_section = cv2.cvtColor(hsv_section, cv2.COLOR_BGR2RGB)

            # Obter a cor média na seção de reconhecimento
            cor_media = np.mean(rgb_section, axis=(0, 1))
            cor_media_tuple = tuple(cor_media.astype(int))

            # Detectar a cor
            cor_nome = convert_rgb_to_names(cor_media_tuple)
            linha_cores.append(cor_nome)

            # Adicionar o nome da cor no quadrado correspondente
            cv2.putText(frame_com_grid, cor_nome, (x_start + 10, y_start + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Adicionar a linha de cores na matriz
        matriz_cores.append(linha_cores)

    # Exibir a matriz de cores no console
    print("Matriz de cores:")
    for linha in matriz_cores:
        print(linha)

    # Exibir o frame com o grid de quadrados no centro e os nomes das cores
    cv2.imshow('Webcam - Reconhecimento de Cores', frame_com_grid)

    # Encerrar ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a webcam e fechar janelas
cap.release()
cv2.destroyAllWindows()
