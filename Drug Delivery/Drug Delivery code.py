from dolfin import *
import numpy as np
from sympy import Heaviside
import math


#Diffusion constant for Drug+ cations GEL
D_Drug = 1.16 * (10**(-6.))

#Diffusion constant for Cl- anioins GEL
D_negative_charges = 1.6 * (10**(-9.))

#Diffusion constant for Drug+ cations SOLUTION
D_Drug_s = 1.16 * (10**(-9.))

#Diffusion constant for Cl- anioins SOLUTION
D_negative_charges_s = 1.6 * (10**(-9.))

#The gas constant
R = 8.31

#Initial value for Drug+ in the solution
c_init_Drug_aqueous_humor = 10

#Initial value for Cl- in the solution
c_init_negative_charges_aqueous_humor = 10


#Initial value for fixed anioin (e.g. bound charge) in the sclera
c_init_Fixed_sclera = 5


#The initial concentration of the drug and negative charges inside the sclera
c_init_Drug_sclera = ((c_init_Fixed_sclera+pow(c_init_Fixed_sclera**2+4*pow(c_init_Drug_aqueous_humor,2),0.5)))/2.
c_init_negative_charges_sclera = ((-c_init_Fixed_sclera+pow(c_init_Fixed_sclera**2+4*pow(c_init_negative_charges_aqueous_humor,2),0.5)))/2.

##############################
#Valence of Drug+
z_Drug = 1.0

#Valence of negative charges
z_negative_charges = -1.0

#Valence of Fixed charges
z_Fixed = -1.0


eps_vacume = 8.85*pow(10,-12)
eps_water = 80.
eps_sclera = 67.

#Dielectric permittivity
epsilon = eps_vacume * eps_water
epsilon_p = eps_vacume * eps_sclera


#Temperature
Temp = 293.0

#Electrical mobility for Drug+ cations
mu_Drug = D_Drug / (R*Temp)
#Electrical mobility for charged ions
mu_negative_charges = D_negative_charges / (R*Temp)


#Faraday number
Faraday = 96485.34

#Reading the mesh
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), "0file.h5", "r")
hdf.read(mesh, "/mesh", False)
subdomains = MeshFunction('size_t', mesh, mesh.topology().dim())


hdf.read(subdomains, "/subdomains")
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
hdf.read(boundaries, "/boundaries")


dx = Measure('dx', domain=mesh, subdomain_data=subdomains)

#Defining element for scalar variables (e.g. concentrations and voltage)
Element1 = FiniteElement("CG", mesh.ufl_cell(), 1)

# Defining the mixed function space
W_elem = MixedElement([Element1, Element1 ,Element1])

W = FunctionSpace(mesh, W_elem)

#Defining the "Trial" functions
z = Function(W)
dz=TrialFunction(W)

#separating the unknown variables
c_Drug, c_negative_charges,  phi = split(z)

#Defining the "test" functions
(v_1, v_2,v_6) = TestFunctions(W)

# Time variables
dt = 1
t = 0
T = 247

#Required function spaces
V_c_Drug = FunctionSpace(mesh, Element1)
V_c_negative_charges = FunctionSpace(mesh, Element1)
V_phi = FunctionSpace(mesh, Element1)

#MeshFunction is defined to be used later for distinguishing between the sclera domain and aqueous humor domain 
materials = MeshFunction("size_t", mesh, mesh.topology().dim())

#The index of the elements corresonding to the sclera domain should be extracted from the mesh file and given here
gel_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ....]

# Defining the initial conditions for the drug
class Drug_initial(UserExpression):

    def __init__(self, **kwargs):
        super().__init__()


        self.c_Drug_sclera = kwargs["c_Drug_sclera"]
        self.c_Drug_aqueous_humor = kwargs["c_Drug_aqueous_humor"]
        self.materials = kwargs["materials"]

    def eval_cell(self, value, x, cell):

        if self.materials[cell.index] in gel_ind:
            value[0] = self.c_Drug_sclera

        else:
            value[0] = self.c_Drug_aqueous_humor

    def value_shape(self):
        return ()

IC_Drug = Drug_initial(materials=materials,c_Drug_sclera = c_init_Drug_sclera, c_Drug_aqueous_humor = c_init_Drug_aqueous_humor, degree=0)

# Previous solution
C_previous_Drug = interpolate(IC_Drug, V_c_Drug)

# Defining the initial conditions for the negative charges
class negative_charges_initial(UserExpression):

    def __init__(self, **kwargs):

        super().__init__()
        self.c_negative_charges_sclera = kwargs["c_negative_charges_sclera"]
        self.c_negative_charges_aqueous_humor = kwargs["c_negative_charges_aqueous_humor"]
        self.materials = kwargs["materials"]


    def eval_cell(self, value, x, cell):

        if self.materials[cell.index] in gel_ind:
            value[0] = self.c_negative_charges_sclera

        else:
            value[0] = self.c_negative_charges_aqueous_humor

    def value_shape(self):
        return ()

IC_negative_charges = negative_charges_initial(materials=materials,c_negative_charges_sclera = c_init_negative_charges_sclera, c_negative_charges_aqueous_humor = c_init_negative_charges_aqueous_humor, degree=0)

# Previous solution
C_previous_negative_charges = interpolate(IC_negative_charges, V_c_negative_charges)



#VaritioDrugl form for the Nernst-Planck equation in the sclera domain for Drug+ and negative charges
#Note that 33 and 34 correspond to the sclera domain and aqueous humor domain
Weak_NP_Drug = c_Drug * v_1 * dx(33) + dt * D_Drug * dot(grad(c_Drug), grad(v_1)) * dx(33) +\
                   dt * z_Drug * mu_Drug * Faraday * c_Drug * dot(grad(phi), grad(v_1)) * dx(33) - C_previous_Drug * v_1 * dx(33)

Weak_NP_negative_charges = c_negative_charges * v_2 * dx(33) + dt * D_negative_charges * dot(grad(c_negative_charges), grad(v_2)) * dx(33) +\
                   dt * z_negative_charges * mu_negative_charges * Faraday * c_negative_charges * dot(grad(phi), grad(v_2)) * dx(33) - C_previous_negative_charges * v_2 * dx(33)


#VaritioDrugl form for the Poisson equation in the sclera domain for Drug+ and Cl- and fixed charge
Weak_Poisson = dot(grad(phi), grad(v_6)) * dx(33) - (Faraday / epsilon_p) * z_Drug * c_Drug * v_6 * dx(33) \
                           - (Faraday / epsilon_p) * z_negative_charges * c_negative_charges * v_6 * dx(33)- (Faraday / epsilon_p) * z_Fixed * c_init_Fixed_sclera * v_6 * dx(33)

##################################

#VaritioDrugl form for the Nernst-Planck equation in the aqueous humor domain for Drug+ and negative charges
Weak_NP_Drug_aqueous_humor = c_Drug * v_1 * dx(34) + dt * D_Drug_s * dot(grad(c_Drug), grad(v_1)) * dx(34) +\
                   dt * z_Drug * mu_Drug * Faraday * c_Drug * dot(grad(phi), grad(v_1)) * dx(34) - C_previous_Drug * v_1 * dx(34)

Weak_NP_negative_charges_aqueous_humor = c_negative_charges * v_2 * dx(34) + dt * D_negative_charges_s * dot(grad(c_negative_charges), grad(v_2)) * dx(34) +\
                   dt * z_negative_charges * mu_negative_charges * Faraday * c_negative_charges * dot(grad(phi), grad(v_2)) * dx(34) - C_previous_negative_charges * v_2 * dx(34)


#VaritioDrugl form for the Poisson equation in the aqueous humor domain for Drug+ and negative charges and fixed charge
Weak_Poisson_aqueous_humor = dot(grad(phi), grad(v_6)) * dx(34) - (Faraday / epsilon) * z_Drug * c_Drug * v_6 * dx(34) \
                           - (Faraday / epsilon) * z_negative_charges * c_negative_charges * v_6 * dx(34)


#Summing up variatioDrugl forms
F = Weak_NP_Drug + Weak_NP_negative_charges + Weak_Poisson + Weak_NP_negative_charges_aqueous_humor + Weak_NP_Drug_aqueous_humor + Weak_Poisson_aqueous_humor

#Defining the boundary conditions
#Note that 35 and 36 correspond to the boundaries where the anode and cathode have been placed on
#W.sub(0)  and W.sub(1) refer to the concentration of the drug and tthe negative charges respectively
bc_left_c = DirichletBC(W.sub(0),Constant(c_init_Drug_sclera), boundaries, 35)
bc_right_c = DirichletBC(W.sub(0),Constant(c_init_Drug_sclera), boundaries, 36)

bc_left_c_1 = DirichletBC(W.sub(1),Constant(c_init_negative_charges_sclera), boundaries, 35)
bc_right_c_1 = DirichletBC(W.sub(1),Constant(c_init_negative_charges_sclera), boundaries, 36)

#Saving the results for the concentration of the drug
q = File("Drug.pvd")


assign(z.sub(1), C_previous_negative_charges)
assign(z.sub(0), C_previous_Drug)

#Solving block
#W.sub(2) refers to the voltage

while t <= T:

    bc_left_volt = DirichletBC(W.sub(2), -1 * Heaviside((t - 37), 0), boundaries, 35)
    bc_right_volt = DirichletBC(W.sub(2), 1 * Heaviside((t - 37), 0), boundaries, 36)
    bcs = [bc_left_volt, bc_right_volt, bc_left_c, bc_right_c, bc_left_c_1, bc_right_c_1 ]

    J = derivative(F, z, dz)

    problem = NonlinearVariatioDruglProblem(F, z, bcs, J)
    solver = NonlinearVariatioDruglSolver(problem)
    solver.parameters['newton_aqueous_humorver']['convergence_criterion'] = 'incremental'
    solver.parameters['newton_aqueous_humorver']['linear_aqueous_humorver'] = 'mumps'
    solver.solve()

    (c_Drug, c_negative_charges,phi) = z.split(True)

    C_previous_Drug.assign(c_Drug)
    C_previous_negative_charges.assign(c_negative_charges)

################################################
    t += dt
    q << c_Drug
