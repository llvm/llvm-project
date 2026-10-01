! Collapsed imperfect loop nests in a target region. OpenMP 6.0 leaves the
! execution count of intervening code unspecified between once per iteration of
! the enclosing loop and once per collapsed logical iteration, so the array
! results here are idempotent and the after-code counter is only range-checked.

! REQUIRES: flang, gpu
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic

module collapse_intervening
  implicit none
contains
  subroutine fill(n, m, a)
    integer, intent(in) :: n, m
    integer, intent(out) :: a(n * m)
    integer :: i, j, offset

    !$omp target teams distribute parallel do collapse(2) private(offset) &
    !$omp   map(from: a)
    do i = 1, n
      offset = (i - 1) * m
      do j = 1, m
        a(offset + j) = i * 100 + j
      end do
    end do
  end subroutine

  ! Intervening code before the loop at two levels, and after the loop at the
  ! outermost level. The after-code is a reduction so that counting it is not a
  ! race.
  subroutine fill3(n, m, p, b, total)
    integer, intent(in) :: n, m, p
    integer, intent(out) :: b(n * m * p)
    integer, intent(out) :: total
    integer :: i, j, k, base, row

    total = 0
    !$omp target teams distribute parallel do collapse(3) private(base, row) &
    !$omp   reduction(+: total) map(from: b)
    do i = 1, n
      base = (i - 1) * m * p
      do j = 1, m
        row = base + (j - 1) * p
        do k = 1, p
          b(row + k) = i * 10000 + j * 100 + k
        end do
      end do
      total = total + 1
    end do
  end subroutine
end module

program target_teams_distribute_collapse_intervening
  use collapse_intervening
  implicit none
  integer :: n, m, p, i, j, k, errors, total
  integer, allocatable :: a(:), b(:)

  ! Runtime values, so the collapsed bounds are host-evaluated.
  n = command_argument_count() + 10
  m = command_argument_count() + 8
  p = command_argument_count() + 6

  allocate(a(n * m), b(n * m * p))
  a = -1
  b = -1

  call fill(n, m, a)
  call fill3(n, m, p, b, total)

  errors = 0
  do i = 1, n
    do j = 1, m
      if (a((i - 1) * m + j) /= i * 100 + j) errors = errors + 1
    end do
  end do

  do i = 1, n
    do j = 1, m
      do k = 1, p
        if (b((i - 1) * m * p + (j - 1) * p + k) /= i * 10000 + j * 100 + k) &
          errors = errors + 1
      end do
    end do
  end do

  if (total < n .or. total > n * m * p) errors = errors + 1

  print *, "number of errors: ", errors
  deallocate(a, b)
  if (errors /= 0) stop 1
end program

! CHECK: number of errors:  0
