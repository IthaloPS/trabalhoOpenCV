import cv2
import numpy as np
import time
from sklearn.neighbors import KDTree
import kociemba

# Dicionário de cores do cubo mágico em RGB
color_dict = {
    "Vermelho": (255, 0, 0),
    "Verde": (0, 255, 0),
    "Azul": (0, 0, 255),
    "Amarelo": (255, 160, 0),
    "Laranja": (255, 130, 0),
    "Branco": (255, 255, 255)
}

# Matriz para cada cor central identificada
matriz_centro = {
    "Vermelho": [],
    "Verde": [],
    "Azul": [],
    "Amarelo": [],
    "Laranja": [],
    "Branco": []
}

array_ordem = []

matriz_to_kociemba = {
    "F":[],
    "R":[],
    "L":[],
    "U":[],
    "B":[],
    "D":[]
}

cor_to_lado = {
    "0": "F",
    "1": "R",
    "2": "B",
    "3": "L",
    "4": "U",
    "5": "D"
}

dicionario_movimentos = {
    "F": "db_imagens/F.png",
    "F'": "db_imagens/FLinha.png",
    "F2": "db_imagens/F2.png",
    "R": "db_imagens/R.png",
    "R'": "db_imagens/RLinha.png",
    "R2": "db_imagens/R2.png",
    "L": "db_imagens/L.png",
    "L'": "db_imagens/LLinha.png",
    "L2": "db_imagens/L2.png",
    "U": "db_imagens/U.png",
    "U'": "db_imagens/ULinha.png",
    "U2": "db_imagens/U2.png",
    "D": "db_imagens/D.png",
    "D'": "db_imagens/DLinha.png",
    "D2": "db_imagens/D2.png",
    "B": "db_imagens/B.png",
    "B'": "db_imagens/BLinha.png",
    "B2": "db_imagens/B2.png"
}


ordem_faces = ['U', 'R', 'F', 'D', 'L', 'B']
cubo = ""

def make_cube():
    cubo = ""
    for face in ordem_faces:
        for linha in matriz_to_kociemba[face]:
            cubo += ''.join(linha)
    return cubo

def transpoe_matriz(matriz):
    for cor, valores in matriz.items():
        novo_cor_to_lado = {array_ordem[i]: valor for i, valor in enumerate(cor_to_lado.values())}
        lado = novo_cor_to_lado[cor]
        matriz_to_kociemba[lado].extend(
            [[novo_cor_to_lado[valor] for valor in sublista] for sublista in valores[0]]
        )

# Contador e armazenamento das verificações para estabilidade
ultima_matriz_verificada = None
contador_estabilidade = 0
ultima_cor_centro = None  # Armazena a última cor de centro salva

# Intervalo de tempo para atualização (em segundos)
intervalo_verificacao = 2
tempo_anterior = time.time()

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
proporcao = 0.6
tamanho_reconhecimento = 0.2

# Inicializar uma matriz vazia para ser usada entre verificações
nova_matriz_cores = []

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Desenhar o grid de quadrados no centro do frame
    frame_com_grid, start_x, start_y, tamanho_linha, tamanho_coluna = desenhar_grid_quadrado(frame.copy(), linhas, colunas, proporcao)

    # Atualizar a matriz de cores a cada intervalo de tempo
    if time.time() - tempo_anterior >= intervalo_verificacao:
        tempo_anterior = time.time()
        
        # Converter para o espaço RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        nova_matriz_cores = []

        # Atualizar matriz de cores detectadas
        for i in range(linhas):
            linha_cores = []
            for j in range(colunas):
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

                # Detectar a cor e adicionar à nova matriz
                cor_nome = convert_rgb_to_names(cor_media_tuple)
                linha_cores.append(cor_nome)
            nova_matriz_cores.append(linha_cores)

        # Print da matriz de cores recém-verificada
        print("Nova matriz de cores verificada:")
        for linha in nova_matriz_cores:
            print(linha)

        # Identificar a cor do centro
        cor_centro_atual = nova_matriz_cores[1][1]

        # Apenas reiniciar se o centro mudou
        if cor_centro_atual != ultima_cor_centro:
            contador_estabilidade = 0
            ultima_cor_centro = cor_centro_atual

        # Verificar estabilidade
        if nova_matriz_cores == ultima_matriz_verificada:
            contador_estabilidade += 1
        else:
            contador_estabilidade = 0
            ultima_matriz_verificada = nova_matriz_cores

        # Se a matriz for constante por 3 verificações, salvar na matriz de centro correspondente
        if contador_estabilidade >= 3:
            matriz_to_save = matriz_centro[cor_centro_atual]
            if(len(matriz_to_save) > 0):
                print(f'Matriz da cor {cor_centro_atual} já foi salva anteriormente')
                print('Posicione o cubo em um centro ainda não salvo')
                print(matriz_centro)
                False
            if cor_centro_atual in matriz_centro and len(matriz_to_save) == 0:
                array_ordem.append(cor_centro_atual)
                matriz_centro[cor_centro_atual].append(nova_matriz_cores)
                print(f"Matriz salva na cor de centro {cor_centro_atual}.")
                print(matriz_centro)
                print('Posicione o cubo em um centro ainda não salvo')
                time.sleep(3)

            # Resetar contador após salvar
            contador_estabilidade = 0

    # Exibir as cores mais recentes na tela, se a matriz estiver preenchida
    if nova_matriz_cores:
        for i in range(linhas):
            for j in range(colunas):
                cor_nome = nova_matriz_cores[i][j]
                x_start = start_x + j * tamanho_coluna
                y_start = start_y + i * tamanho_linha
                cv2.putText(frame_com_grid, cor_nome, (x_start + 10, y_start + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if(all(len(lista) > 0 for lista in matriz_centro.values())):

        transpoe_matriz(matriz_centro)
        print(matriz_to_kociemba)
        cubo = make_cube()
        #print(cubo)
        solve = kociemba.solve(cubo)
        movimentos = solve.split()

        pagina_html = "<!DOCTYPE html>\n<html>\n<head>\n<title>Resolução do Cubo Mágico</title>\n</head>\n<body>\n"
        pagina_html += "<h1>Instruções para Resolver o Cubo Mágico</h1>\n"

        for mov in movimentos:
            if mov in dicionario_movimentos:
                pagina_html += f'<div style="margin-bottom: 20px;">\n'
                pagina_html += f'<h3>{mov}</h3>\n'
                pagina_html += f'<img src="{dicionario_movimentos[mov]}" alt="{mov}" style="width: 200px; height: 200px;">\n'
                pagina_html += '</div>\n'

        pagina_html += "</body>\n</html>"

        with open("resolucao_cubo.html", "w") as file:
            file.write(pagina_html)

        print("Página HTML gerada com sucesso: resolucao_cubo.html") 
        break


    # Exibir o frame com o grid de quadrados e nomes das cores
    cv2.imshow('Webcam - Reconhecimento de Cores', frame_com_grid)

    # Encerrar ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a webcam e fechar janelas
cap.release()
cv2.destroyAllWindows()
