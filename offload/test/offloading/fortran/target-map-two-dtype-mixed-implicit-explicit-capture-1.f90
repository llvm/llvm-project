! Offloading test checking interaction of two
! derived type's with a mix of explicit and
! implicit member mapping to target
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: scalar_array
        real(4) :: break_0
        real(4) :: array_x(10)
        real(4) :: break_1
        real(4) :: array_y(10)
        real(4) :: break_3
    end type scalar_array

    type(scalar_array) :: scalar_arr1
    type(scalar_array) :: scalar_arr2

  !$omp target map(tofrom:scalar_arr1%break_1)
    scalar_arr2%break_3 = 10
    scalar_arr1%break_1 = 15
  !$omp end target

  print*, scalar_arr1%break_1
  print*, scalar_arr2%break_3
end program main

!CHECK: 15.
!CHECK: 10.
