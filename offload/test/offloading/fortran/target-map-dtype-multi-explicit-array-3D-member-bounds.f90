! Offloading test checking interaction of an
! explicit derived type member mapping of
! two arrays with explicit bounds when
! mapped to target
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: scalar_array
    real(4) :: break_0
    integer(4) :: array_x(3,3,3)
    real(4) :: break_1
    integer(4) :: array_y(3,3,3)
    real(4) :: break_3
    end type scalar_array

    type(scalar_array) :: scalar_arr

    do i = 1, 3
      do j = 1, 3
        do k = 1, 3
            scalar_arr%array_x(i, j, k) = 42
            scalar_arr%array_y(i, j, k) = 0 ! Will get overwritten by garbage values in target
        end do
       end do
    end do

  !$omp target map(tofrom:scalar_arr%array_x(1:3, 1:3, 2:2), scalar_arr%array_y(1:3, 1:3, 1:3))
    do j = 1, 3
      do k = 1, 3
        scalar_arr%array_y(k, j, 2) = scalar_arr%array_x(k, j, 2)
      end do
    end do
  !$omp end target

  print *, scalar_arr%array_y
end program main

!CHECK: 0 0 0 0 0 0 0 0 0 42 42 42 42 42 42 42 42 42 0 0 0 0 0 0 0 0
