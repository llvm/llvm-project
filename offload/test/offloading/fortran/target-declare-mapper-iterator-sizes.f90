! Offloading test checking the per-iteration size of iterated declare mapper
! entries: an element of a multi-dimensional component (one bound per
! dimension) and a two-element section (upper bound above lower bound). The
! kernel writes every mapped element, so an undersized entry would leave some
! of them unchanged on the host.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program target_declare_mapper_iterator_sizes
  implicit none
  integer, parameter :: rows = 3, cols = 4, nc = 8

  type :: grid_t
    integer :: a(rows, cols)
    integer :: c(nc)
  end type grid_t

  ! Row 2 of every column of `a`, and `c` in adjacent pairs.
  !$omp declare mapper(m: grid_t :: v) &
  !$omp&  map(iterator(i = 1:cols), tofrom: v%a(2, i)) &
  !$omp&  map(iterator(j = 1:nc/2), tofrom: v%c(2*j-1 : 2*j))

  type(grid_t) :: x
  integer :: i, j
  logical :: ok

  x%a = 0
  x%c = 0

  !$omp target map(mapper(m): x)
    do i = 1, cols
      x%a(2, i) = 100 + i
    end do
    do j = 1, nc
      x%c(j) = 200 + j
    end do
  !$omp end target

  ok = .true.
  do i = 1, cols
    if (x%a(2, i) /= 100 + i) ok = .false.
  end do
  ! Rows 1 and 3 were not mapped and must be untouched.
  do i = 1, cols
    if (x%a(1, i) /= 0 .or. x%a(3, i) /= 0) ok = .false.
  end do
  do j = 1, nc
    if (x%c(j) /= 200 + j) ok = .false.
  end do

  if (ok) then
    print *, "Test passed!"
  else
    print *, "Test failed!"
    print *, x%a
    print *, x%c
  end if
end program target_declare_mapper_iterator_sizes

! CHECK: Test passed!
