import numpy as np
import pickle
import argparse

import resource
import signal
import tracemalloc

def skew_sym(vec):
    return np.array([[0, -vec[2], vec[1]],
                     [vec[2], 0, -vec[0]],
                     [-vec[1], vec[0], 0]])

def rotation_to_vector(R) -> np.ndarray:
    theta = np.arccos((np.trace(R)-1)/2)

    # TODO: How to solve axis
    U, S, Vt = np.linalg.svd(R - np.eye(3)) # better not use solve (Rx = x)
    axis = Vt[-1, :]
    axis /= np.linalg.norm(axis)
    
    # TODO: R is trivial rotation?
    
    return theta*axis # psi

# Rodrigues'
def vector_to_rotation(v) -> np.ndarray:
    theta = np.linalg.norm(v)
    if theta < 1e-6:
        return np.eye(3)
    axis = v/theta
    # print(axis.shape)
    # exit()
    
    skew = skew_sym(axis)
    
    # TODO: Projection to SO(3) with SVD, is this necessary step?
    R = np.eye(3) + (1-np.cos(theta))*skew@skew + np.sin(theta)*skew
    
    # U, _, Vt = np.linalg.svd(R)
    # R_proj = U @ Vt
    # if np.linalg.det(R_proj) < 0:
    #     Vt[-1, :] *= -1
    #     R_proj = U @ Vt
    return R



def pack_params(problem) -> np.ndarray:
    params = []
    n_cameras = problem['poses'].shape[0]
    calib_flag = problem['is_calibrated']
    for i in range(n_cameras):
        R = problem['poses'][i,:3,:3]
        r = rotation_to_vector(R)
        t = problem['poses'][i,:3, 3]
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
    errors = np.zeros((len(problem['observations']) * 2))
    
    for i, obs in enumerate(problem['observations']):
        cam_id, point_id, x, y = obs
        X = data['points'][point_id]
        P = data['poses'][cam_id]
        R, t = P[:,:3], P[:,3]
        f = data['focal_lengths'][cam_id] if not problem['is_calibrated'] else problem['focal_lengths'][cam_id]
        X_Cam_coord = R @ X + t
        Z = X_Cam_coord[2]
        
        # TODO: determine the criterion for small value, 1e-5? 1e-6?
        
        if Z < 1e-6:
            errors[2*i:2*i+2] = 0
            continue
        x_proj = f * X_Cam_coord[0] / Z
        y_proj = f * X_Cam_coord[1] / Z

        errors[2*i] = x_proj - x
        errors[2*i+1] = y_proj - y
    
    return errors

def lstsq(error):
    return 0.5*np.sum(error**2)

def compute_jacobian(problem, data):
    """
    problem (dict): contains the problem definition
    data (dict): contains the current state of the problem
    """
    n_cameras = problem['poses'].shape[0]
    n_points = problem['points'].shape[0]
    cam_param_size = 6 if problem['is_calibrated'] else 7
    # blocks
    
    # TODO: Using dict to store block diagonal matrices, is this efficient enough?
    A = {i: np.zeros((cam_param_size, cam_param_size)) for i in range(n_cameras)}
    B = {i: np.zeros((3, 3)) for i in range(n_points)}
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
        if Z_prime < 1e-6:
            continue
        inv_Z = 1 / Z_prime
        inv_Z2 = inv_Z * inv_Z
        
        J_c = np.zeros((2, cam_param_size))
        J_p = np.zeros((2, 3))
        
        # How to compute small jacobian blocks?
        # Compare jacobian with numerical jacobian
        residual = np.array([f * X_prime * inv_Z - x ,f * Y_prime * inv_Z - y]) 
        
        proj_jacobian = f * np.array([[inv_Z, 0, - X_prime * inv_Z2],
                                  [0, inv_Z, - Y_prime * inv_Z2]])
        
        focal_jacobian = np.zeros((3, 1))
        
        if not problem['is_calibrated']:
            focal_jacobian = np.array([[X_prime * inv_Z], [Y_prime * inv_Z], [0]])
        
        translation_jacobian = np.eye(3)
        
        point_jacobian = R
        
        rotation_jacobian = - skew_sym(R @ X) # why?
        
        cam = np.hstack([rotation_jacobian, translation_jacobian])
        
        if not problem['is_calibrated']:
            cam = np.hstack([cam, focal_jacobian])
        
        J_c = (proj_jacobian @ cam)
        J_p = (proj_jacobian @ point_jacobian)
        
        A[cam_id] += J_c.T @ J_c
        B[point_id] += J_p.T @ J_p
        C_key = (cam_id, point_id)
        C[C_key] = C.get(C_key, np.zeros((cam_param_size, 3))) + J_c.T @ J_p
        
        JTr_cam[cam_id * cam_param_size : (cam_id+1)*cam_param_size] -= J_c.T @ residual
        JTr_point[point_id * 3 : (point_id+1)*3] -= J_p.T @ residual
        
    return A, B, C, JTr_cam, JTr_point
        
def shur_complement(problem, A, B, C, JTr_cam, JTr_point, lambda_):
    """    Compute the Schur complement of the block matrix

    Args:
        A (_type_): block diagonal matrices, (N, cam_param_size, cam_param_size)
        B (_type_): block diagonal matrices, (M, 3, 3)
        C (_type_): dense matrices, (cam_param_size, 3)
    """
    n_cameras = problem['poses'].shape[0]
    cam_param_size = 6 if problem['is_calibrated'] else 7
    B_inv = {}
    for point_id in B.keys():
        B_inv[point_id] = np.linalg.inv(B[point_id] + lambda_ * np.eye(3)) # numerical stability
    # compute S = A - C * B^{-1} * C^T
    S = A.copy()
    for (cam_id, point_id), C_block in C.items():
        # print(B_inv.keys(), point_id, cam_id)
        B_inv_block = B_inv[point_id]
        update = C_block @ B_inv_block @ C_block.T
        S[cam_id] -= update
        
    for cam_id in S:
        diag = np.diag(S[cam_id]).copy()
        np.fill_diagonal(S[cam_id], diag + (lambda_ + 1e-8) * np.ones_like(diag))
        
    # solve for delta
    delta_cam = list()
    
    for cam_id in range(n_cameras):
        start_idx = cam_id * cam_param_size
        end_idx = start_idx + cam_param_size
        delta = np.linalg.solve(S[cam_id], JTr_cam[start_idx:end_idx])
        delta_cam.append(delta)
    
    delta_cam = np.concatenate(delta_cam)
    
    delta_point = np.zeros_like(JTr_point)
    
    # cam_param_size = 6 if problem['is_calibrated'] else 7
    
    for (cam_id, point_id), C_block in C.items():
        start = point_id * 3
        end = start + 3
        delta_point[start:end] = B_inv[point_id] @ (JTr_point[start:end] - C_block.T @ delta_cam[cam_id * cam_param_size: (cam_id + 1) * cam_param_size])
    
    return np.concatenate([delta_cam, delta_point])

def LM_Sparse(problem, max_iter = 100):
    params = pack_params(problem)
    lambda_ = 1e-4
    prev_cost = np.inf
    obs = len(problem['observations'])
    
    for _ in range(max_iter):
        data = unpack_params(params, problem)
        errors = reprojection_error(params, problem)
        cost = 0.5*np.sum(errors**2)
        msqe = np.sum(np.sqrt(errors**2))/obs
        
        # TODO: What is the condition for convergence? The mean loss, the norm of the update? How to select proper delta
        if  msqe <= 0.1:
            print("Converged")
            break
        if cost >= prev_cost:
            lambda_ *= 2
        else:
            lambda_ *= 0.5
            prev_cost = cost
        A, B, C, JTr_cam, JTr_points = compute_jacobian(problem, data)
        delta = shur_complement(problem, A, B, C, JTr_cam, JTr_points, lambda_)
        
        # TODO: Choice of learning rate
        update = delta * 0.1
        
        if np.linalg.norm(update) < 1e-5:
            print("Converged")
            break
        params += update
        if _ % 100 == 0:
            validate_rotations(unpack_params(params, problem)['poses'], iter = _)
        print(f"iter: {_}, cost: {msqe}, update: {np.linalg.norm(update)}")
    return unpack_params(params, problem)            

def validate_rotations(poses, iter = 1000):
    for cam_id, cam_pose in enumerate(poses):
        R = cam_pose[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
            print(f"iter {iter} Camera {cam_id}: Invalid rotation matrix: R^T R ≠ I")
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-5):
            print(f"iter {iter} Camera {cam_id}:Invalid rotation matrix: det(R) ≠ 1")
    return True

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
    
    # print(f"problem: {problem['poses'].shape} \n", f"points: {problem['points'].shape} \n", f"focal_lengths: {problem['focal_lengths'].shape} \n")
    # print(f"is_calibrated: {problem['is_calibrated']} \n")
    solution['observations'] = sorted(solution['observations'], key=lambda x: (x[1], x[0]))
    optimize = LM_Sparse(problem)
    
    for key in optimize.keys():
        if key == 'focal_lengths':
            solution[key] = optimize[key]
        elif key == 'points':
            solution[key] = optimize[key]
        elif key == 'poses':
            solution[key] = optimize[key]
    validate_rotations(solution['poses'])
    
    return solution

def set_max_runtime(ms):
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (ms, hard))
    signal.signal(signal.SIGXCPU, exit_function)


def limit_memory(maxbytes):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxbytes, hard))
    signal.signal(signal.SIGXCPU, exit_function)
    
def exit_function(signo, frame):
    raise SystemExit(1)

if __name__ == '__main__':
    # set_max_runtime(600)

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', help="config file")
    args = parser.parse_args()

    problem = pickle.load(open(args.problem, 'rb'))
    
    limit_memory(1073741824) # 1GB
    
    solution = solve_ba_problem(problem)

    solution_path = args.problem.replace(".pickle", "-solution.pickle")
    pickle.dump(solution, open(solution_path, "wb"))
