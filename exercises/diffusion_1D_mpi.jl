# 1D linear diffusion Julia MPI solver
using Plots, Printf, MAT
import MPI

# enable plotting by default
if !@isdefined do_save; do_save = true end

# MPI functions
@views function update_halo(A, neighbors_x, comm)
    # Send to / receive from neighbor 1 ("left neighbor")

    # Send to / receive from neighbor 2 ("right neighbor")
    
    return
end

@views function diffusion_1D_mpi(; do_save=false)
    # MPI
    MPI.Init()
    dims        = [0]
    comm        = MPI.COMM_WORLD
    nprocs      = MPI.Comm_size(comm)
    MPI.Dims_create!(nprocs, dims)
    comm_cart   = MPI.Cart_create(comm, dims, [0], 1)
    me          = MPI.Comm_rank(comm_cart)
    coords      = MPI.Cart_coords(comm_cart)
    neighbors_x = MPI.Cart_shift(comm_cart, 0, 1)
    if (me==0) println("nprocs=$(nprocs), dims[1]=$(dims[1])") end
    # Physics
    lx    = 10.0
    λ     = 1.0
    nt    = 100
    # Numerics
    nx    = 32                 # local number of grid points
    nx_g  = dims[1]*(nx-2) + 2 # global number of grid points
    # Derived numerics
    dx    = lx/nx_g            # global
    dt    = dx^2/λ/2.1
    # Array allocation
    qHx   = zeros(nx-1)
    # Initial condition
    x0    = coords[1]*(nx-2)*dx
    xc    = [x0 + ix*dx - dx/2 - 0.5*lx  for ix=1:nx]
    H     = exp.(.-xc.^2)
    t_tic = 0.0
    # Time loop
    for it = 1:nt
        if (it==11) t_tic = Base.time() end
        qHx        .= .-λ*diff(H)/dx
        H[2:end-1] .= H[2:end-1] .- dt*diff(qHx)/dx
        update_halo(H, neighbors_x, comm_cart)
    end
    t_toc = Base.time()-t_tic
    if (me==0) @printf("Time = %1.4e s, T_eff = %1.2f GB/s \n", t_toc, round((2/1e9*nx*sizeof(lx))/(t_toc/(nt-10)), sigdigits=2)) end
    if do_save file = matopen("$(@__DIR__)/H_$(me).mat", "w"); write(file, "H", Array(H)); close(file) end
    MPI.Finalize()
    return
end

diffusion_1D_mpi(; do_save=do_save)
