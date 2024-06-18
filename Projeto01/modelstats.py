import torch

def get_model_stats(model):
    """Retorna estatísticas dos parâmetros de um modelo neural.
    
    Para cada parâmetro, retorna o maior e menor valor do parâmetro
    e o maior e menor valor do gradiente do parâmetro.
    """
    stats = {}
    for param_name, param in model.named_parameters():
        # Valores do parâmetro
        param_max = param.max().item()
        param_min = param.min().item()
        
        # Valores do gradiente
        if param.grad is not None:
            grad_max = param.grad.max().item()
            grad_min = param.grad.min().item()
        else:
            grad_max = None
            grad_min = None
        
        stats[param_name] = {
            'param_max': param_max,
            'param_min': param_min,
            'grad_max': grad_max,
            'grad_min': grad_min
        }
    
    return stats

# # Passa a imagem pelo modelo para calcular os scores
# scores = model(img)
# # Calcula os gradientes (em um treinamento real seria loss.backward())
# scores.sum().backward()

# # Obtém as estatísticas dos parâmetros
# stats = get_model_stats(model)

def get_overall_stats(stats):
    """Retorna o menor e maior valor dentre todos os parâmetros e gradientes.

    Args:
        stats (dict): Dicionário com estatísticas dos parâmetros e gradientes.

    Returns:
        dict: Dicionário com os menores e maiores valores.
    """
    # Inicializa variáveis para armazenar os valores mínimos e máximos
    overall_param_min = float('inf')
    overall_param_max = float('-inf')
    overall_grad_min = float('inf')
    overall_grad_max = float('-inf')
    
    for _,values in stats.items():
        # Atualiza os valores mínimos e máximos dos parâmetros
        overall_param_min = min(overall_param_min, values['param_min'])
        overall_param_max = max(overall_param_max, values['param_max'])
        
        # Atualiza os valores mínimos e máximos dos gradientes, se disponíveis
        if values['grad_min'] is not None:
            overall_grad_min = min(overall_grad_min, values['grad_min'])
        if values['grad_max'] is not None:
            overall_grad_max = max(overall_grad_max, values['grad_max'])
    
    return overall_param_min, overall_param_max, overall_grad_min, overall_grad_max
    
@torch.no_grad()
def get_four_stats(model):
    
    stats=get_model_stats(model)
    return get_overall_stats(stats)

