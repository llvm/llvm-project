! Offloading test checking interaction of an
! explicit derived type member mapping of
! an array when mapped to target
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

   type(scalar_array) :: scalar_arr

  !$omp target map(tofrom:scalar_arr%array_y)
    do i = 1, 10
      scalar_arr%array_y(i) = i
    end do
  !$omp end target

  print *, scalar_arr%array_y
end program main

!CHECK: 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.
