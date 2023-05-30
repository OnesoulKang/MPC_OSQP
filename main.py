import gymnasium as gym
import numpy as np
from scipy import sparse
import osqp 

env = gym.make("Pendulum-v1", render_mode='human')
'''
The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |
'''
s, _ = env.reset()
th = np.arccos(s[0])

dt = env.dt
g = env.g
l = env.l
m = env.m

Ad = sparse.csc_matrix([[        1, dt], 
                        [3*g/(2*l)*dt, 1]])
Bd = sparse.csc_matrix([0, 3*dt/(m*l*l)]).transpose()
[nx, nu] = Bd.shape

u0 = 0
umin = np.array([-10])
umax = -umin

xmin = np.array([-np.pi, -8])
xmax = -xmin

Q = sparse.diags([1, 0.1])
QN = Q
R = 0.01 * sparse.eye(nu)

# x0 = np.zeros(2)
x0 = np.array([th, s[2]])
xr = np.array([0, 0])

N = 100

P = sparse.block_diag([sparse.kron(np.eye(N), Q), QN, 
                       sparse.kron(np.eye(N), R)], format='csc')

q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
               np.zeros(N*nu)])

# - linear dynamics
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])

ueq = leq
# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, warm_start=False, verbose=False)

for x in range(1000):
    # breakpoint()
    env.render()
    res = prob.solve()
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')
    x = res.x[-N*nu:-(N-1)*nu]
    # tau = -env.m * env.g * env.l * s[1]
    s, _, _,  _, _ = env.step(x)

    # Update initial state
    th = np.arccos(s[0])
    print(th)
    x0 = np.array([th, s[2]])
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)

env.close()