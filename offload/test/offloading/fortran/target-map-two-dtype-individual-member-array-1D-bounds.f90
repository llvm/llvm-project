! Offloading test checking interaction of two
! derived type's with a single explicit array
! member each being mapped with bounds to
! target
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
    real(4) :: array_x(10)
    real(4) :: break_1
    real(4) :: array_y(10)
    real(4) :: break_3
    end type scalar_array

    type(scalar_array) :: scalar_arr1
    type(scalar_array) :: scalar_arr2


  !$omp target map(tofrom:scalar_arr1%array_x(3:6), scalar_arr2%array_x(3:6))
    do i = 3, 6
      scalar_arr2%array_x(i) = i
      scalar_arr1%array_x(i) = i
    end do
  !$omp end target

  print*, scalar_arr1%array_x
  print*, scalar_arr2%array_x
end program main

!CHECK: 0. 0. 3. 4. 5. 6. 0. 0. 0. 0.
!CHECK: 0. 0. 3. 4. 5. 6. 0. 0. 0. 0.
