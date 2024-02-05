! Offloading test checking interaction of allocatables
! with target in different scopes
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
module test
    contains
  subroutine func_arg(arg_alloc) 
    integer,  allocatable, intent (inout) :: arg_alloc(:)
  
  !$omp target map(tofrom: arg_alloc) 
    do index = 1, 10
      arg_alloc(index) = arg_alloc(index) + index
    end do
  !$omp end target
    
    print *, arg_alloc
  end subroutine func_arg
end module 
  
subroutine func 
    integer,  allocatable :: local_alloc(:) 
    allocate(local_alloc(10))
  
  !$omp target map(tofrom: local_alloc) 
    do index = 1, 10
      local_alloc(index) = index
    end do
  !$omp end target
  
    print *, local_alloc
  
    deallocate(local_alloc)
end subroutine func 
  
  
program main 
  use test 
  integer,  allocatable :: map_ptr(:) 
  
  allocate(map_ptr(10))
  
  !$omp target map(tofrom: map_ptr) 
    do index = 1, 10
      map_ptr(index) = index
    end do
  !$omp end target

   call func 

   print *, map_ptr

   call func_arg(map_ptr)

   deallocate(map_ptr)
end program 

! CHECK: 1 2 3 4 5 6 7 8 9 10
! CHECK: 1 2 3 4 5 6 7 8 9 10
! CHECK: 2 4 6 8 10 12 14 16 18 20
