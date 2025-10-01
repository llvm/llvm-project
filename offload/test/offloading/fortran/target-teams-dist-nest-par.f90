! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic
! RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic
program main
  integer :: array(10) = 0
  integer :: x, y, z
  !$omp target
  !$omp teams distribute private(x, y)
  OuterLoopOne: do x=1,1
     array(2) = 42
     OuterLoopTwo: do y=1,1
        !$omp parallel do private(z)
        InnerLoopOne: do z=1,10
           array(z) = 20
        enddo InnerLoopOne
        !$omp end parallel do
     enddo OuterLoopTwo
  enddo OuterLoopOne
  !$omp end teams distribute
  !$omp end target
  ! Expected to print all 20's
  print *, array
end program main

! CHECK: 20 20 20 20 20 20 20 20 20 20
