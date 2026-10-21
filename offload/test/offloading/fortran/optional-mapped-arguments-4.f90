! Implicitly mapping an absent optional assumed-shape argument must produce a
! complete descriptor even when a runtime flag guards all accesses to it.
! Also check present arguments and a higher-rank descriptor with explicit
! lower bounds.
! REQUIRES: flang, amdgpu
! RUN: %libomptarget-compile-fortran-generic -O0
! RUN: %libomptarget-run-generic | %fcheck-generic
! RUN: %libomptarget-compile-fortran-generic -O2
! RUN: %libomptarget-run-generic | %fcheck-generic

module optional_arguments
  implicit none
  logical :: use_b = .false.
  !$omp declare target(use_b)
contains
  subroutine rank_one(a, n, b)
    integer, intent(in) :: n
    real(8), intent(inout) :: a(n)
    real(8), optional, intent(in) :: b(:)
    integer :: i

    !$omp target teams distribute parallel do
    do i = 1, n
      if (use_b) then
        a(i) = a(i) + b(2)
      else
        a(i) = a(i) + 1
      end if
    end do
  end subroutine

  subroutine rank_five(a, n, b)
    integer, intent(in) :: n
    real(8), intent(inout) :: a(n)
    real(8), optional, intent(in) :: b(-2:, 0:, 4:, -1:, 3:)
    integer :: i

    !$omp target teams distribute parallel do
    do i = 1, n
      if (use_b) then
        a(i) = a(i) + b(-1, 1, 5, 0, 4)
      else
        a(i) = a(i) + 1
      end if
    end do
  end subroutine
end module

program main
  use optional_arguments
  implicit none
  integer, parameter :: n = 100
  real(8) :: a(n), b(2), c(2, 2, 2, 2, 2)

  a = 0
  b = [2d0, 3d0]
  c = -1000
  c(2, 2, 2, 2, 2) = 7

  !$omp target data map(tofrom: a)
  call rank_one(a, n)
  call rank_five(a, n)
  !$omp end target data
  if (any(a /= 2)) stop 1

  use_b = .true.
  !$omp target update to(use_b)
  !$omp target data map(tofrom: a)
  call rank_one(a, n, b)
  call rank_five(a, n, c)
  !$omp end target data
  if (any(a /= 12)) stop 2

  print *, "PASSED"
end program

! CHECK: PASSED
