! Offloading test checking a declare mapper whose only map clause uses an
! iterator modifier. Every element of the array component is mapped by the
! iterator; the kernel writes elements other than the first, which only
! round-trips if the per-element entries are grouped under the mapper
! variable (MEMBER_OF) rather than treated as independent objects.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program target_declare_mapper_iterator
  implicit none
  integer, parameter :: n = 8

  type :: vec_t
    integer :: a(n)
  end type vec_t

  !$omp declare mapper(m: vec_t :: v) map(iterator(i = 1:n), tofrom: v%a(i))

  type(vec_t) :: x
  integer :: i
  logical :: ok

  x%a = 0

  !$omp target map(mapper(m): x)
    do i = 1, n
      x%a(i) = 10 * i
    end do
  !$omp end target

  ok = .true.
  do i = 1, n
    if (x%a(i) /= 10 * i) ok = .false.
  end do

  if (ok) then
    print *, "Test passed!"
  else
    print *, "Test failed!"
    print *, x%a
  end if
end program target_declare_mapper_iterator

! CHECK: Test passed!
