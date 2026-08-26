! Test that host-evaluated clause operands are still available when the
! referenced scalar is also privatized on an inner leaf of a composite target
! construct. Indicating that we have not optimized the implicit map out due
! to privatization.
!
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
! XFAIL: intelgpu

program main
  implicit none
  integer :: errors

  errors = 0

  call check_parallel_do_num_threads(4, errors)
  call check_teams_num_teams(4, errors)

  print *, "errors:", errors

contains
  subroutine check_parallel_do_num_threads(n, errors)
    use omp_lib
    implicit none
    integer :: n
    integer, intent(inout) :: errors
    integer :: observed
    integer :: i

    observed = -1
    !$omp target parallel do private(n) num_threads(n) map(tofrom: observed)
    do i = 1, 1
      observed = omp_get_num_threads()
      n = 99
    end do
    !$omp end target parallel do

    if (observed .ne. n) errors = errors + 1
  end subroutine

  subroutine check_teams_num_teams(n, errors)
    use omp_lib
    implicit none
    integer :: n
    integer, intent(inout) :: errors
    integer :: observed
    integer :: i

    observed = -1
    !$omp target teams distribute private(n) num_teams(n) map(tofrom: observed)
    do i = 1, 1
      observed = omp_get_num_teams()
      n = 99
    end do
    !$omp end target teams distribute

    if (observed .ne. n) errors = errors + 1
  end subroutine
end program main

! CHECK: errors: 0
