! Offloading test checking interaction of an
! enter and exit map of an scalar
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    integer :: scalar
    scalar = 10

    !$omp target enter data map(to: scalar)
    !ignored, as we've already attached
    scalar = 20

   !$omp target
      scalar = scalar + 50
   !$omp end target

  !$omp target exit data map(from: scalar)

  ! not the answer one may expect, but it is the same
  ! answer Clang gives so we are correctly on par with
  ! Clang for the moment.
  print *, scalar
end program

!CHECK: 10
