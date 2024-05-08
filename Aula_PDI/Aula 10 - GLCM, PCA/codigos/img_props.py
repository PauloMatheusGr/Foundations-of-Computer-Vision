import numpy as np
import scipy.ndimage as ndi

def get_area(img):
    '''Calcula a área do objeto contido em img'''
    
    return np.sum(img)

def get_perimeter(img):
    '''Calcula o perímetro do objeto contido em img utilizando contagem simples
       de pixels'''
    
    elem_est = np.ones((3, 3))
    img_eroded = ndi.binary_erosion(img, elem_est)
    img_border = img - img_eroded
    perimeter = np.sum(img_border)    
        
    return perimeter

def get_circularity(img):
    '''Calcula circularidade do objeto contido em img'''
    
    area = get_area(img)
    perimeter = get_perimeter(img)
    
    circularity = 4*np.pi*area/perimeter**2
    
    return circularity    

def get_centroid(img):
    '''Calcula a posição do centróide do objeto contido em img'''
    
    pixel_indices = np.nonzero(img==1)
    avg_row = np.mean(pixel_indices[0])
    avg_col = np.mean(pixel_indices[1])
       
    return avg_row, avg_col
    
def get_distance_to_centroid(img):
    '''Calcula a distância entre cada ponto de contorno e
       o centróide do objeto contido em img'''
    
    avg_row, avg_col = get_centroid(img)
    contour = image_contour(img)        
    contour = np.array(contour)
    dist_to_centroid = np.sqrt((contour[:,0]-avg_row)**2 + (contour[:,1]-avg_col)**2)
        
    return dist_to_centroid

def image_contour(img):
    '''Obtém o contorno paramétrico de um objeto contido
       no array img.'''
    
    # Mapeamento utilizado para encontrar o vizinho inicial a ser
    # buscado na próxima iteração dado o vizinho do ponto atual
    # Por exemplo, se o ponto atual for (12, 15) e o próximo ponto
    # de borda for (12, 16), isso significa que o vizinho de índice
    # 2 será o próximo ponto de borda. Nesse novo ponto, precisamos
    # buscar a partir do vizinho de índice 1, pois o vizinho de
    # índice 0 foi o último ponto a ser verificado antes de encontrarmos
    # o ponto atual
    neighbor_map = [7, 7, 1, 1, 3, 3, 5, 5]
    
    # Adiciona 0 ao redor da imagem para evitar pontos 
    # do objeto tocando a borda
    img_pad = np.pad(img, 1, mode='constant')
    
    num_rows, num_cols = img_pad.shape
    k = 0
    row = 0
    col = 0
    # Busca do primeiro ponto do objeto
    while img_pad[row, col]==0:
        k += 1
        row = k//num_cols
        col = k - row*num_cols
        
    curr_point = (row, col)    # Ponto atual
    contour = [curr_point]     # Pontos do contorno
    starting_index = 2         # Índice do vizinho inicial a ser verificado
    while True:
        next_point, last_index = get_next_point(img_pad, curr_point, 
                                                    starting_index)

        # Novo índice do vizinho inicial baseado no último
        # índice buscado
        starting_index = neighbor_map[last_index]
        
        # Critério de parada. Se o ponto adicionado na iteração anterior (contour[-1])
        # for o mesmo que o primeiro ponto (contour[0]) e o ponto atual for o mesmo
        # que o segundo ponto adicionado, o algoritmo termina. Só podemos fazer essa
        # verificação se o contorno possuir ao menos 2 pontos. Ou seja, nosso algoritmo
        # não está tratando o caso de um objeto com apenas 1 pixel
        if len(contour)>1:
            if next_point==contour[1] and contour[-1]==contour[0]:
                break
                
        contour.append(next_point)
        curr_point = next_point
        
    # Subtrai 1 de cada ponto pois o contorno foi encontrado para a
    # imagem preenchida com 0 na borda
    for point_index, point in enumerate(contour):
        contour[point_index] = (point[0]-1, point[1]-1)
        
    return contour

def get_next_point(img, curr_point, starting_index):
    '''Encontra o próximo ponto de borda dado um ponto
       corrente curr_point e o índice do primeiro vizinho
       a ser verificado (starting_index)'''
     
    # Lista dos pontos vizinhos dado o índice do vizinho
    nei_list = [(-1,0), (-1,1), (0,1), (1,1), 
                (1,0), (1,-1),(0,-1), (-1,-1)]
    
    curr_index = starting_index
    nei_value = 0
    while nei_value==0:
        nei_shift = nei_list[curr_index]
        nei_row = curr_point[0] + nei_shift[0]
        nei_col = curr_point[1] + nei_shift[1]
        nei_value = img[nei_row, nei_col]
        if nei_value==1:
            return (nei_row, nei_col), curr_index
        else:
            curr_index = (curr_index+1)%8

def get_distance_to_centroid_props(dist_to_centroid):
    '''Obtém algumas propriedades de um objeto a partir de uma lista de
       distâncias entre os pontos de contorno e o centróide do mesmo'''
    
    avg_dist_to_centroid = np.mean(dist_to_centroid)
    max_dist_to_centroid = np.max(dist_to_centroid)/avg_dist_to_centroid
    min_dist_to_centroid = np.min(dist_to_centroid)/avg_dist_to_centroid
    std_dist_to_centroid = np.std(dist_to_centroid)/avg_dist_to_centroid
    
    return max_dist_to_centroid, min_dist_to_centroid, std_dist_to_centroid
