! Offloading test checking a declare mapper whose iterated element carries its
! own (nested) mapper, applied to an array of the outer type. The outer mapper
! runs once per array element; for each iteration the inner mapper must be
! invoked for the selected component element.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program target_declare_mapper_iterator_nested
  implicit none
  integer, parameter :: ninner = 3, nouter = 2

  type :: inner_t
    integer :: val
    integer :: pad
  end type inner_t

  type :: outer_t
    type(inner_t) :: items(ninner)
  end type outer_t

  !$omp declare mapper(mi: inner_t :: w) map(tofrom: w%val)
  !$omp declare mapper(mo: outer_t :: v) &
  !$omp&  map(mapper(mi), iterator(i = 1:ninner), tofrom: v%items(i))

  type(outer_t) :: arr(nouter)
  integer :: i, k
  logical :: ok

  do k = 1, nouter
    do i = 1, ninner
      arr(k)%items(i)%val = 0
      arr(k)%items(i)%pad = -1
    end do
  end do

  !$omp target map(mapper(mo): arr)
    do k = 1, nouter
      do i = 1, ninner
        arr(k)%items(i)%val = 10 * k + i
      end do
    end do
  !$omp end target

  ok = .true.
  do k = 1, nouter
    do i = 1, ninner
      if (arr(k)%items(i)%val /= 10 * k + i) ok = .false.
      ! `pad` is not mapped by the inner mapper and must be untouched.
      if (arr(k)%items(i)%pad /= -1) ok = .false.
    end do
  end do

  if (ok) then
    print *, "Test passed!"
  else
    print *, "Test failed!"
    do k = 1, nouter
      print *, arr(k)%items(:)%val
    end do
  end if
end program target_declare_mapper_iterator_nested

! CHECK: Test passed!
