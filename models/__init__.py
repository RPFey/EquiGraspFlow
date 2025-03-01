import torch

from utils.distributions import get_dist
from models.vn_dgcnn import VNDGCNNEncoder
from models.vector_fields import VectorField4SE3
from utils.ode_solvers import get_ode_solver
from models.equi_grasp_flow import EquiGraspFlow


def get_model(cfg_model):
    name = cfg_model.pop('name')
    checkpoint = cfg_model.get('checkpoint', None)

    if name == 'equigraspflow':
        model = get_equigraspflow(cfg_model)
    else:
        raise NotImplementedError(f"Model {name} is not implemented.")
    
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location='cpu')

        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])

    return model


def get_equigraspflow(cfg):
    p_uncond = cfg.pop('p_uncond')
    guidance = cfg.pop('guidance')

    init_dist = get_dist(cfg.pop('init_dist'))
    encoder = get_net(cfg.pop('encoder'))
    vector_field = get_net(cfg.pop('vector_field'))
    ode_solver = get_ode_solver(cfg.pop('ode_solver'))

    model = EquiGraspFlow(p_uncond, guidance, init_dist, encoder, vector_field, ode_solver)

    return model


def get_net(cfg_net):
    name = cfg_net.pop('name')

    if name == 'vn_dgcnn_enc':
        net = get_vn_dgcnn_enc(cfg_net)
    elif name == 'vector_field_SE3':
        net = get_vector_field_SE3(cfg_net)
    else:
        raise NotImplementedError(f"Network {name} is not implemented.")
    
    return net


def get_vn_dgcnn_enc(cfg):
    net = VNDGCNNEncoder(**cfg)

    return net


def get_vector_field_SE3(cfg):
    net = VectorField4SE3(**cfg)

    return net
