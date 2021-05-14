import sys
import numpy as np
from scipy import optimize as opt
import gmsh
from mpi4py import MPI as nMPI
import meshio
import dolfin
# We set up the mesh generation and finite element solution as prt of the 
# objective function Cost(x)

def Cost(xp):
    comm = nMPI.COMM_WORLD
    mpi_rank = comm.Get_rank()

    x1, x2 = xp #The two variables (length and feed offset)

    rs = 8.0  # radiation boundary radius
    l = x1  # Patch length
    w = 4.5  # Patch width
    s1 = x2 * x1 / 2.0  # Feed offset
    h = 1.0   # Patch height
    t = 0.05   # Metal thickness
    lc = 1.0  # Coax length
    rc = 0.25 # Coax shield radius
    cc = 0.107 #Coax center conductor 50 ohm air diel
    eps = 1.0e-4
    tol = 1.0e-6
    eta = 377.0 # vacuum intrinsic wave impedance
    eps_c = 1.0 # dielectric permittivity

    k0 = 2.45 * 2.0 * np.pi / 30.0 # Frequency in GHz
    ls = 0.025 #Mesh density parameters for GMSH
    lm = 0.8
    lw = 0.06
    lp = 0.3

    # Run GMSH only on one MPI processor (process 0).
    # We use the GMSH Python interface to generate the geometry and mesh objects
    if mpi_rank == 0:
        print("x[0] = {0:<f}, x[1] = {1:<f} ".format(xp[0], xp[1]))
        print("length = {0:<f}, width = {1:<f}, feed offset = {2:<f}".format(l, w, s1))
        gmsh.initialize()
        gmsh.option.setNumber('General.Terminal', 1)
        gmsh.model.add("SimplePatchOpt")
# Radiation sphere
        gmsh.model.occ.addSphere(0.0, 0.0, 0.0, rs, 1)
        gmsh.model.occ.addBox(0.0, -rs, 0.0, rs, 2*rs, rs, 2)
        gmsh.model.occ.intersect([(3,1)],[(3,2)], 3, removeObject=True, removeTool=True)
# Patch
        gmsh.model.occ.addBox(0.0, -l/2, h, w/2, l, t, 4)
# coax center
        gmsh.model.occ.addCylinder(0.0, s1, -lc, 0.0, 0.0, lc+h, cc, 5, 2.0*np.pi)

# coax shield
        gmsh.model.occ.addCylinder(0.0, s1, -lc, 0.0, 0.0, lc, rc, 7)
        gmsh.model.occ.addBox(0.0, s1-rc, -lc, rc, 2.0*rc, lc, 8)
        gmsh.model.occ.intersect([(3,7)], [(3,8)], 9, removeObject=True, removeTool=True)
        gmsh.model.occ.fuse([(3,3)], [(3,9)], 10, removeObject=True, removeTool=True)
# cutout internal boundaries
        gmsh.model.occ.cut([(3,10)], [(3,4),(3,5)], 11, removeObject=True, removeTool=True)

        gmsh.option.setNumber('Mesh.MeshSizeMin', ls)
        gmsh.option.setNumber('Mesh.MeshSizeMax', lm)
        gmsh.option.setNumber('Mesh.Algorithm', 6)
        gmsh.option.setNumber('Mesh.Algorithm3D', 1)
        gmsh.option.setNumber('Mesh.MshFileVersion', 4.1)
        gmsh.option.setNumber('Mesh.Format', 1)
        gmsh.option.setNumber('Mesh.MinimumCirclePoints', 36)
        gmsh.option.setNumber('Mesh.CharacteristicLengthFromCurvature', 1)

        gmsh.model.occ.synchronize()

        pts = gmsh.model.getEntities(0)
        gmsh.model.mesh.setSize(pts, lm) #Set background mesh density
        pts = gmsh.model.getEntitiesInBoundingBox(-eps, -l/2-eps, h-eps, w/2+eps, l/2+eps, h+t+eps)
        gmsh.model.mesh.setSize(pts, ls)

        pts = gmsh.model.getEntitiesInBoundingBox(-eps, s1-rc-eps, -lc-eps, rc+eps, s1+rc+eps, h+eps)
        gmsh.model.mesh.setSize(pts, lw)
        pts = gmsh.model.getEntitiesInBoundingBox(-eps, -rc-eps, -eps, rc+eps, rc+eps, eps)
        gmsh.model.mesh.setSize(pts, lw)

# Embed points to reduce mesh density on patch faces
        fce1 = gmsh.model.getEntitiesInBoundingBox(-eps, -l/2-eps, h+t-eps, w/2+eps, l/2+eps, h+t+eps, 2)
        gmsh.model.occ.synchronize()
        gmsh.model.geo.addPoint(w/4, -l/4, h+t, lp, 1000)
        gmsh.model.geo.addPoint(w/4, 0.0, h+t, lp, 1001)
        gmsh.model.geo.addPoint(w/4, l/4, h+t, lp, 1002)
        gmsh.model.geo.synchronize()
        gmsh.model.occ.synchronize()
        print(fce1)
        fce2 = gmsh.model.getEntitiesInBoundingBox(-eps, -l/2-eps, h-eps, w/2+eps, l/2+eps, h+eps, 2)
        gmsh.model.geo.addPoint(w/4, -9*l/32, h, lp, 1003)
        gmsh.model.geo.addPoint(w/4, 0.0, h, lp, 1004)
        gmsh.model.geo.addPoint(w/4, 9*l/32, h, lp, 1005)
        gmsh.model.geo.synchronize()
        for tt in fce1:
           gmsh.model.mesh.embed(0, [1000, 1001, 1002], 2, tt[1])
        for tt in fce2:
           gmsh.model.mesh.embed(0, [1003, 1004, 1005], 2, tt[1])
        print(fce2)
        gmsh.model.occ.remove(fce1)
        gmsh.model.occ.remove(fce2)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [11], 1)
        gmsh.model.setPhysicalName(3, 1, "Air")
        gmsh.model.mesh.optimize("Relocate3D", niter=5)
        gmsh.model.mesh.generate(3)
        gmsh.write("SimplePatch.msh")
        gmsh.finalize()
# Mesh generation is finished.  We now use Meshio to translate GMSH mesh to xdmf file for 
# importation into Fenics FE solver
        msh = meshio.read("SimplePatch.msh")
        for cell in msh.cells:
            if  cell.type == "tetra":
                tetra_cells = cell.data

        for key in msh.cell_data_dict["gmsh:physical"].keys():
            if key == "tetra":
                tetra_data = msh.cell_data_dict["gmsh:physical"][key]

        tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells},
                           cell_data={"VolumeRegions":[tetra_data]})

        meshio.write("mesh.xdmf", tetra_mesh)
# Here we import the mesh into Fenics
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile("mesh.xdmf") as infile:
        infile.read(mesh)
    mvc = dolfin.MeshValueCollection("size_t", mesh, 3)
    with dolfin.XDMFFile("mesh.xdmf") as infile:
        infile.read(mvc, "VolumeRegions")
    cf = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc)
# The boundary classes for the FE solver
    class PEC(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    class InputBC(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[2], -lc, tol)

    class OutputBC(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            rr = np.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
            return on_boundary and dolfin.near(rr, 8.0, 1.0e-1)

    class PMC(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dolfin.near(x[0], 0.0, tol)


# Volume domains
    dolfin.File("VolSubDomains.pvd").write(cf)
    dolfin.File("Mesh.pvd").write(mesh)
# Mark boundaries
    sub_domains = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(4)
    pec = PEC()
    pec.mark(sub_domains, 0)
    in_port = InputBC()
    in_port.mark(sub_domains, 1)
    out_port = OutputBC()
    out_port.mark(sub_domains, 2)
    pmc = PMC()
    pmc.mark(sub_domains, 3)
    dolfin.File("BoxSubDomains.pvd").write(sub_domains)
# Set up function spaces
    cell = dolfin.tetrahedron
    ele_type = dolfin.FiniteElement('N1curl', cell, 2, variant="integral") # H(curl) element for EM
    V2 = dolfin.FunctionSpace(mesh, ele_type * ele_type)
    V = dolfin.FunctionSpace(mesh, ele_type)
    (u_r, u_i) = dolfin.TrialFunctions(V2)
    (v_r, v_i) = dolfin.TestFunctions(V2)
    dolfin.info(mesh)
#surface integral definitions from boundaries
    ds = dolfin.Measure('ds', domain = mesh, subdomain_data = sub_domains)
#volume regions
    dx_air = dolfin.Measure('dx', domain = mesh, subdomain_data = cf, subdomain_id = 1)
    dx_subst = dolfin.Measure('dx', domain = mesh, subdomain_data = cf, subdomain_id = 2)
# with source and sink terms
    u0 = dolfin.Constant((0.0, 0.0, 0.0)) #PEC definition
# The incident field sources (E and H-fields)
    h_src = dolfin.Expression(('-(x[1] - s) / (2.0 * pi * (pow(x[0], 2.0) + pow(x[1] - s,2.0)))', 'x[0] / (2.0 * pi *(pow(x[0],2.0) + pow(x[1] - s,2.0)))', 0.0), degree = 2,  s = s1)
    e_src = dolfin.Expression(('x[0] / (2.0 * pi * (pow(x[0], 2.0) + pow(x[1] - s,2.0)))', 'x[1] / (2.0 * pi *(pow(x[0],2.0) + pow(x[1] - s,2.0)))', 0.0), degree = 2, s = s1)
    Rrad = dolfin.Expression(('sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])'), degree = 2)
#Boundary condition dictionary
    boundary_conditions = {0: {'PEC' : u0},
                       1: {'InputBC': (h_src)},
                       2: {'OutputBC': Rrad}}

    n = dolfin.FacetNormal(mesh)

#Build PEC boundary conditions for real and imaginary parts
    bcs = []
    for i in boundary_conditions:
        if 'PEC' in boundary_conditions[i]:
            bc = dolfin.DirichletBC(V2.sub(0), boundary_conditions[i]['PEC'], sub_domains, i)
            bcs.append(bc)
            bc = dolfin.DirichletBC(V2.sub(1), boundary_conditions[i]['PEC'], sub_domains, i)
            bcs.append(bc)

# Build input BC source term and loading term
    integral_source = []
    integrals_load =[]
    for i in boundary_conditions:
        if 'InputBC' in boundary_conditions[i]:
            r = boundary_conditions[i]['InputBC']
            bb1 = 2.0 * (k0 * eta) * dolfin.inner(v_i, dolfin.cross(n, r)) * ds(i) #Factor of two from field equivalence principle
            integral_source.append(bb1)
            bb2 = dolfin.inner(dolfin.cross(n, v_i), dolfin.cross(n, u_r)) * k0 * np.sqrt(eps_c) * ds(i)
            integrals_load.append(bb2)
            bb2 = dolfin.inner(-dolfin.cross(n, v_r), dolfin.cross(n, u_i)) * k0 * np.sqrt(eps_c) * ds(i)
            integrals_load.append(bb2)

    for i in boundary_conditions:
        if 'OutputBC' in boundary_conditions[i]:
           r = boundary_conditions[i]['OutputBC']
           bb2 = (dolfin.inner(dolfin.cross(n, v_i), dolfin.cross(n, u_r)) * k0 + 1.0 * dolfin.inner(dolfin.cross(n, v_i), dolfin.cross(n, u_i)) / r)* ds(i)
           integrals_load.append(bb2)
           bb2 = (dolfin.inner(-dolfin.cross(n, v_r), dolfin.cross(n, u_i)) * k0 + 1.0 * dolfin.inner(dolfin.cross(n, v_r), dolfin.cross(n, u_r)) / r)* ds(i)
           integrals_load.append(bb2)
# for PMC, do nothing. Natural BC.

    a = (dolfin.inner(dolfin.curl(v_r), dolfin.curl(u_r)) + dolfin.inner(dolfin.curl(v_i), dolfin.curl(u_i)) - eps_c * k0 * k0 * (dolfin.inner(v_r, u_r) + dolfin.inner(v_i, u_i))) * dx_subst + (dolfin.inner(dolfin.curl(v_r), dolfin.curl(u_r)) + dolfin.inner(dolfin.curl(v_i), dolfin.curl(u_i)) - k0 * k0 * (dolfin.inner(v_r, u_r) + dolfin.inner(v_i, u_i))) * dx_air + sum(integrals_load)
    L = sum(integral_source)

    u1 = dolfin.Function(V2)
    vdim = u1.vector().size()
    print("Solution vector size =", vdim)

    dolfin.solve(a == L, u1, bcs, solver_parameters = {'linear_solver' : 'mumps'}) 

#Here we write files of the field solution for inspection
    u1_r, u1_i = u1.split(True)
    fp = dolfin.File("EField_r.pvd")
    fp << u1_r
    fp = dolfin.File("EField_i.pvd")
    fp << u1_i
# Compute power relationships and reflection coefficient
    H = dolfin.interpolate(h_src, V) # Get input field
    P =  dolfin.assemble((-dolfin.dot(u1_r,dolfin.cross(dolfin.curl(u1_i),n))+dolfin.dot(u1_i,dolfin.cross(dolfin.curl(u1_r),n))) * ds(2))
    P_refl = dolfin.assemble((-dolfin.dot(u1_i,dolfin.cross(dolfin.curl(u1_r), n)) + dolfin.dot(u1_r, dolfin.cross(dolfin.curl(u1_i), n))) * ds(1))
    P_inc = dolfin.assemble((dolfin.dot(H, H) * eta / (2.0 * np.sqrt(eps_c))) * ds(1))
    print("Integrated power on port 2:", P/(2.0 * k0 * eta))
    print("Incident power at port 1:", P_inc)
    print("Integrated reflected power on port 1:", P_inc - P_refl / (2.0 * k0 * eta))
#Reflection coefficient is returned as cost function
    rho_old = (P_inc - P_refl / (2.0 * k0 * eta)) / P_inc #Fraction of incident power reflected as objective function
    return rho_old

#Optimization
x0 = np.array([5.0, 0.675]) # Starting point for optimization
res = opt.minimize(Cost, x0, method='Nelder-Mead', options={'maxiter':100, 'disp':True, 'fatol':0.003})
print(res)
sys.exit(0)



