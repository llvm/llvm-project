! Offloading test checking interaction of an
! explicit derived type mapping when mapped to 
! target and assigning to individual members
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: scalar
    integer(4) :: ix = 0
    real(4) :: rx = 0.0
    complex(4) :: zx = (0,0)
    integer(4) :: array(5)
    end type scalar 
  
    type(scalar) :: out
    type(scalar) :: in
  
    in%ix = 10
    in%rx = 2.0
    in%zx = (2, 10)
  
    do i = 1, 5
      in%array(i) = i
    end do 
  
  !$omp target map(from:out) map(to:in)
    out%ix = in%ix
    out%rx = in%rx
    out%zx = in%zx
  
    do i = 1, 5
      out%array(i) = in%array(i)
    end do 
  !$omp end target
  
    print*, in%ix
    print*, in%rx
    print*, in%array
    write (*,*) in%zx

    print*, out%ix
    print*, out%rx
    print*, out%array
    write (*,*)  out%zx
end program main

!CHECK: 10
!CHECK: 2.
!CHECK: 1 2 3 4 5
!CHECK: (2.,10.)
!CHECK: 10
!CHECK: 2.
!CHECK: 1 2 3 4 5
!CHECK: (2.,10.)
