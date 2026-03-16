! REQUIRES: flang, libc
! RUN: %libomptarget-compile-fortran-run-and-check-generic

program io_test
  implicit none

  integer :: i
  real :: r
  complex :: c
  logical :: l

  i = 42
  r = 3.14
  c = (1.0, -1.0)
  l = .true.

  ! CHECK: Text 42 3.14 (1.,-1.) T
  ! CHECK: Text 42 3.14 (1.,-1.) T
  ! CHECK: Text 42 3.14 (1.,-1.) T
  ! CHECK: Text 42 3.14 (1.,-1.) T
  !$omp target teams num_teams(4)
  !$omp parallel num_threads(1)
    print *, "Text", " ", i, " ", r, " ", c, " ", l
  !$omp end parallel
  !$omp end target teams

end program io_test
