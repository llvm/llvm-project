! Offloading test checking interaction of an
! explicit derived type mapping when mapped
! to target and assinging one derived type
! to another
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
    end type scalar

    type(scalar) :: in
    type(scalar) :: out
    in%ix = 10
    in%rx = 2.0
    in%zx = (2, 10)

  !$omp target map(from:out) map(to:in)
      out = in
  !$omp end target

    print*, in%ix
    print*, in%rx
    write (*,*) in%zx

    print*, out%ix
    print*, out%rx
    write (*,*)  out%zx
end program main

!CHECK: 10
!CHECK: 2.
!CHECK: (2.,10.)
!CHECK: 10
!CHECK: 2.
!CHECK: (2.,10.)
