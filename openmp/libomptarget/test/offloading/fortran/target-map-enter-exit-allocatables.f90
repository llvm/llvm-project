! Offloading test checking interaction of allocatables
! with enter, exit and target 
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    integer, allocatable :: A(:)    
    allocate(A(10))
    
   !$omp target enter data map(alloc: A)

    !$omp target
        do I = 1, 10
            A(I) = I
        end do
    !$omp end target

    !$omp target exit data map(from: A)

    !$omp target exit data map(delete: A)
    
    do i = 1, 10
        print *, A(i)
    end do
    
    deallocate(A)
end program

! CHECK: 1
! CHECK: 2
! CHECK: 3
! CHECK: 4
! CHECK: 5
! CHECK: 6
! CHECK: 7
! CHECK: 8
! CHECK: 9
! CHECK: 10
