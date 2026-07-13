! Check that we do not have a multiple definition issue when 
! compiling multiple files declaring a common block and declare 
! target and that we resolve to the correct value. 
! Slightly trivialized variation of a coding pattern in 
! larger Fortran applications like Nekbone and SPEC benchmarks.
! REQUIRES: flang, amdgpu

! RUN: %flang -c -fopenmp --offload-arch=gfx90a \
! RUN:   %S/../../Inputs/declare-target-common-block-sub.f90 -o declare-target-common-block-sub.o
! RUN: %libomptarget-compile-fortran-generic declare-target-common-block-sub.o
! RUN: %t | %fcheck-generic

program main
  common /dxyz/ arr(10)
  !$omp declare target (/dxyz/)
  arr = 0.0
  call foo()
  print *, arr(1)
end program

! CHECK: 1.
