! Basic offloading test with a target region
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
  integer :: x;
  x = 0
!$omp target map(from:x)
    x = 5
!$omp end target
  print *, "x = ", x
end program main

! CHECK: x = 5

