! Offloading test checking interaction of an
! single explicit member map from a single
! derived type.
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    real :: test
    type :: scalar
        integer(4) :: ix = 0
        real(4) :: rx = 0.0
        complex(4) :: zx = (0,0)
        real(4) :: ry = 1.0
    end type scalar

    type(scalar) :: scalar_struct
    scalar_struct%rx = 2.0
    test = 21.0

  !$omp target map(from:scalar_struct%rx)
    scalar_struct%rx = test
  !$omp end target

  print *, scalar_struct%rx
end program main

!CHECK: 21.
