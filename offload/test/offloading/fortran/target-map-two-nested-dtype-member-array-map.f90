! Offloading test checking interaction of an
! explicit member map utilising array bounds
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: array
     real(4) :: array_z(10)
     real(4) :: break_4
     real(4) :: array_ix(10)
    end type array

    type :: scalar_array
    real(4) :: break_0
    real(4) :: array_x(10)
    real(4) :: break_1
    real(4) :: array_y(10)
    real(4) :: break_3
    type(array) :: nested
    end type scalar_array

    type(scalar_array) :: scalar_arr1
    type(scalar_array) :: scalar_arr2

  do i = 1, 10
    scalar_arr1%nested%array_z(i) = i
    scalar_arr2%nested%array_z(i) = i
  end do

  !$omp target map(tofrom:scalar_arr1%nested%array_z(3:6), scalar_arr1%nested%array_ix(3:6), scalar_arr2%nested%array_z(3:6), scalar_arr2%nested%array_ix(3:6))
    do i = 3, 6
      scalar_arr2%nested%array_ix(i) = scalar_arr1%nested%array_z(i)
    end do

    do i = 3, 6
      scalar_arr1%nested%array_ix(i) = scalar_arr2%nested%array_z(i)
    end do
  !$omp end target

  print*, scalar_arr1%nested%array_ix
  print*, scalar_arr2%nested%array_z

  print*, scalar_arr2%nested%array_ix
  print*, scalar_arr1%nested%array_z
end program main

!CHECK: 0. 0. 3. 4. 5. 6. 0. 0. 0. 0.
!CHECK: 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.
!CHECK: 0. 0. 3. 4. 5. 6. 0. 0. 0. 0.
!CHECK: 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.
