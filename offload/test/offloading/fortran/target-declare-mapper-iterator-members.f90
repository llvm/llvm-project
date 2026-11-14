! Offloading test checking a declare mapper that maps other members of the
! variable explicitly, plus elements of an array component via an iterator
! modifier, without mapping the variable itself. The parent component for the
! iterated elements must be the first one the mapper pushes; if it came after
! the explicit members, the runtime would reject it as an extension of their
! mapping.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program target_declare_mapper_iterator_members
  implicit none
  integer, parameter :: n = 4

  type :: rec_t
    integer :: tag
    integer :: a(n)
    integer :: b
  end type rec_t

  !$omp declare mapper(m: rec_t :: v) map(tofrom: v%tag, v%b) &
  !$omp&  map(iterator(i = 1:n), tofrom: v%a(i))

  type(rec_t) :: x
  integer :: i
  logical :: ok

  x%tag = 1
  x%b = 2
  x%a = 0

  !$omp target map(mapper(m): x)
    x%tag = x%tag * 10
    x%b = x%b * 10
    do i = 1, n
      x%a(i) = x%tag + x%b + i
    end do
  !$omp end target

  ok = (x%tag == 10) .and. (x%b == 20)
  do i = 1, n
    if (x%a(i) /= 30 + i) ok = .false.
  end do

  if (ok) then
    print *, "Test passed!"
  else
    print *, "Test failed!"
    print *, x%tag, x%b, x%a
  end if
end program target_declare_mapper_iterator_members

! CHECK: Test passed!
