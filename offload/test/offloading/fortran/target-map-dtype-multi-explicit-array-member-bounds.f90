! Offloading test checking interaction of an
! explicit derived type member mapping of
! two arrays with explicit bounds when
! mapped to target
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

    type(scalar_array) :: scalar_arr

  do i = 1, 10
    scalar_arr%array_x(i) = i
  end do

  !$omp target map(tofrom:scalar_arr%array_x(3:6), scalar_arr%array_y(3:6))
    do i = 1, 10
      scalar_arr%array_y(i) = scalar_arr%array_x(i)
    end do
  !$omp end target

  print*, scalar_arr%array_y
end program main

!CHECK: 0. 0. 3. 4. 5. 6. 0. 0. 0. 0.
