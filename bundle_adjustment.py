from operator import inv
import numpy as np
import pickle
import argparse

def rotation_to_vector(R) -> np.ndarray:
    theta = np.arccos((np.trace(R)-1)/2)
    axis = np.linalg.solve(R-np.identity(3),np.zeros(3))
    axis /= np.linalg.norm(axis)
    
    return theta*axis # psi

# Rodrigues'
def vector_to_rotation(v) -> np.ndarray:
    theta = np.linalg.norm(v)
    if theta < 1e-6:
        return np.eye(3)
    axis = v/theta
    
    skew = np.array([[0, -axis[2], axis[1]],
                     axis[2], 0, -axis[1],
                     -axis[1], axis[0], 0])
    
    R = np.eye(3) + (1-np.cos(theta))*v@v.T + np.sin(theta)*skew
    
    return R

# TODO: consider sparse structure

def pack_params(problem) -> np.ndarray:
    params = []
    n_cameras = problem['poses'].shape[0]
    calib_flag = problem['is_calibrated']
    for i in range(n_cameras):
        R = problem['poses'][i,:,:3]
        r = rotation_to_vector(R)
        t = problem['poses'][i,:,3]
        params.extend(r)
        params.extend(t)
        if not calib_flag:
            params.append(problem['focal_lengths'][i])
    params.extend(problem['points'].flatten())
    return np.array(params)
    # [ksi, f, points]
    # ksi: [psi, t] R^6
    # points: [x, y, z] R^3, M points
    
def unpack_params(params, problem):
    calib_flag = problem['is_calibrated']
    n_cameras = problem['poses'].shape[0]
    n_points = problem['points'].shape[0]
    
    poses = []
    focal_lengths = []
    
    idx = 0
    
    for _ in range(n_cameras):
        r = params[idx: idx + 3]
        idx += 3
        t = params[idx: idx + 3]
        idx += 3
        R = vector_to_rotation(r)
        pose = np.hstack([R, t.reshape(-1,1)])
        poses.append(pose)
        if not calib_flag:
            focal_lengths.append(params[idx])
            idx += 1
    points = params[idx: idx + n_points*3].reshape(-1,3)
    
    return {
        'poses': np.array(poses),
        'points': points,
        'focal_lengths': np.array(focal_lengths) if not calib_flag else problem['focal_lengths']
    }

# reprojection error for each observation
def reprojection_error(params, problem):
    data = unpack_params(params, problem)
    errors = []
    
    for obs in problem['observations']:
        cam_id, point_id, x, y = obs
        X = data['points'][point_id]
        P = data['poses'][cam_id]
        R, t = P[:,:3], P[:,3]
        f = data['focal_lengths'][cam_id] if not problem['is_calibrated'] else problem['focal_lengths'][cam_id]
        X_Cam_coord = R @ X + t
        Z = X_Cam_coord[2]
        
        # TODO: determine the criterion for small value, 1e-5? 1e-6?
        
        if Z < 1e-6:
            errors.extend([0.0, 0.0])
            continue
        x_proj = f * X_Cam_coord[0] / Z
        y_proj = f * X_Cam_coord[1] / Z
        error_x = x - x_proj
        error_y = y - y_proj
        
        errors.extend([error_x, error_y])
    
    return np.array(errors)

def compute_jacobian(problem, data):
    n_cameras = problem['poses'].shape[0]
    n_points = problem['points'].shape[0]
    cam_param_size = 6 if not problem['is_calibrated'] else 7
    # blocks
    A = {i: np.zeros((cam_param_size, cam_param_size)) for i in range(n_cameras)}
    B = {i: np.zeros((3, 3)) for i in range(n_cameras)}
    C = {} # dense
    
    """
    | A   C |
    |       |
    | C^T B |
    """
    
    JTr_cam = np.zeros(n_cameras * cam_param_size)
    JTr_point = np.zeros(n_points * 3)
    
    for obs_idx, (cam_id, point_id, x, y) in enumerate(problem['observations']):
        X = data['points'][point_id]
        P = data['poses'][cam_id]
        R, t = P[:,:3], P[:,3]
        f = data['focal_lengths'][cam_id] if not problem['is_calibrated'] else problem['focal_lengths'][cam_id]
        X_Cam_coord = R @ X + t
        X_prime, Y_prime, Z_prime = X_Cam_coord
        
        # for stability
        # TODO: check if Z_prime is small?
        inv_Z = 1 / Z_prime
        inv_Z2 = inv_Z * inv_Z
        
        J_c = np.zeros((2, cam_param_size))
        J_p = np.zeros((2, 3))
        
        # How to compute small jacobian blocks?
        
        A[cam_id] += J_c.T @ J_c
        B[cam_id] += J_p.T @ J_p
        C_key = (cam_id, point_id)
        C[C_key] = ...
        
    
    
    return A, B, C, JTr_cam, JTr_point
        
def shur_complement(problem, A, B, C, JTr_cam, JTr_point, lambda_):
    """    Compute the Schur complement of the block matrix

    Args:
        A (_type_): block diagonal matrices, (N, cam_param_size, cam_param_size)
        B (_type_): block diagonal matrices, (M, 3, 3)
        C (_type_): dense matrices, (cam_param_size, 3)
    """
    
    B_inv = {}
    for point_id in B:
        B_inv[point_id] = np.linalg.inv(B[point_id] + lambda_ * np.eye(3)) # B + lambda*I
    # compute S = A - C * B^{-1} * C^T
    S = A.copy()
    for (cam_id, point_id), C_block in C.items():
        B_inv_block = B_inv[point_id]
        update = C_block @ B_inv_block @ C_block.T
        S[cam_id] -= update
        
    for cam_id in S:
        diag = np.diag(S[cam_id]) # ?
        diag *= (1 + lambda_)
        np.fill_diagonal(S[cam_id], diag)
        
    # solve for delta
    delta_cam = np.linalg.solve(S, JTr_cam)
    
    delta_point = np.zeros_like(JTr_point)
    
    cam_param_size = 6 if not problem['is_calibrated'] else 7
    
    for (cam_id, point_id), C_block in C.items():
        start = point_id * 3
        end = start + 3
        delta_point[start:end] = B_inv[point_id] @ (JTr_point[start:end] - C_block.T @ delta_cam[cam_id * cam_param_size: (cam_id + 1) * cam_param_size])
    
    return np.concatenate([delta_cam, delta_point])

def LM_Sparse(problem, max_iter = 100):
    params = pack_params(problem)
    lambda_ = 1e-3
    prev_cost = np.inf
    
    for _ in range(max_iter):
        data = unpack_params(params, problem)
        errors = reprojection_error(params, problem)
        cost = 0.5 * np.sum(errors**2)
        if cost >= prev_cost:
            lambda_ *= 2
        else:
            lambda_ *= 0.8
            prev_cost = cost
    A, B, C, JTr_cam, JTr_points = compute_jacobian(problem, data)
            



def solve_ba_problem(problem):
    '''
    Solves the bundle adjustment problem defined by "problem" dict

    Input:
        problem: bundle adjustment problem containing the following fields:
            - is_calibrated: boolean, whether or not the problem is calibrated
            - observations: list of (cam_id, point_id, x, y)
            - points: [n_points,3] numpy array of 3d points
            - poses: [n_cameras,3,4] numpy array of camera extrinsics
            - focal_lengths: [n_cameras] numpy array of focal lengths
    Output:
        solution: dictionary containing the problem, with the following fields updated
            - poses: [n_cameras,3,4] numpy array of optmized camera extrinsics
            - points: [n_points,3] numpy array of optimized 3d points
            - (if is_calibrated==False) then focal lengths should be optimized too
                focal_lengths: [n_cameras] numpy array with optimized focal focal_lengths

    Your implementation should optimize over the following variables to minimize reprojection error
        - problem['poses']
        - problem['points']
        - problem['focal_lengths']: if (is_calibrated==False)

    '''

    solution = problem
    # YOUR CODE STARTS


    return solution



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', help="config file")
    args = parser.parse_args()

    problem = pickle.load(open(args.problem, 'rb'))
    solution = solve_ba_problem(problem)

    solution_path = args.problem.replace(".pickle", "-solution.pickle")
    pickle.dump(solution, open(solution_path, "wb"))
