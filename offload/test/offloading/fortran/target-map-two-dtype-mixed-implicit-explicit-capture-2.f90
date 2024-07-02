! Offloading test checking interaction of two
! derived type's with a mix of explicit and
! implicit member mapping of arrays to target
! one with bounds.
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

  do i = 1, 10
    scalar_arr1%array_x(i) = i
  end do

  !$omp target map(tofrom:scalar_arr2%array_x(3:6))
    do i = 3, 6
      scalar_arr2%array_x(i) = scalar_arr1%array_x(i)
    end do
  !$omp end target

  print*, scalar_arr1%array_x
  print*, scalar_arr2%array_x
end program main

!CHECK: 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.
!CHECK: 0. 0. 3. 4. 5. 6. 0. 0. 0. 0.
