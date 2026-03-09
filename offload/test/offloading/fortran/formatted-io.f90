! REQUIRES: flang, libc
! RUN: %libomptarget-compile-fortran-run-and-check-generic

program formatted_io_test
  implicit none

  integer :: i, ios
  real :: r
  complex :: c
  logical :: l
  character(len=5) :: s

  i = 42
  r = 3.14
  c = (1.0, -1.0)
  l = .true.
  s = "Hello"

  ! CHECK: i=  42
  !$omp target
    write(*, '(A,I4)') "i=", i
  !$omp end target

  ! CHECK: r= 3.14
  !$omp target
    write(*, '(A,F5.2)') "r=", r
  !$omp end target

  ! CHECK: s=Hello
  !$omp target map(to : s)
    write(*, '(A,A)') "s=", s
  !$omp end target

  ! CHECK: l=T
  !$omp target
    write(*, '(A,L1)') "l=", l
  !$omp end target

  ! CHECK: c=(  1.00, -1.00)
  !$omp target
    write(*, '(A,A,F6.2,A,F6.2,A)') "c=", "(", real(c), ",", aimag(c), ")"
  !$omp end target

  ! CHECK: mixed: Hello   42 3.14 T
  !$omp target map(to : s)
    write(*, '(A,A,I5,F5.2,L2)') "mixed: ", s, i, r, l
  !$omp end target

  ! Test IOSTAT handling
  ! CHECK: iostat:           0
  !$omp target
    write(*, '(A)', iostat=ios) "dummy"
    write(*, '(A,I12)') "iostat:", ios
  !$omp end target

  ! Test formatted output from multiple teams.
  ! CHECK: val=  42
  ! CHECK: val=  42
  ! CHECK: val=  42
  ! CHECK: val=  42
  !$omp target teams num_teams(4)
  !$omp parallel num_threads(1)
    write(*, '(A,I4)') "val=", i
  !$omp end parallel
  !$omp end target teams

end program formatted_io_test
