! Offloading test checking interaction of an derived type mapping of two 
! explicit members to target
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: scalar
      integer(4) :: ix = 0
      real(4) :: rx = 0.0
      complex(4) :: zx = (0,0)
      real(4) :: ry = 1.0
    end type scalar

      type(scalar) :: scalar_struct

    !$omp target map(from:scalar_struct%rx, scalar_struct%ry)
      scalar_struct%rx = 21.0
      scalar_struct%ry = 27.0
    !$omp end target

    print*, scalar_struct%rx
    print*, scalar_struct%ry
end program main

!CHECK: 21.
!CHECK: 27.
