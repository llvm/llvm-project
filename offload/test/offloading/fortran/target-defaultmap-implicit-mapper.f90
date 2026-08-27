! Offload test that ensures defaultmap(tofrom: scalar) does not suppress
! implicit default mapper generation for allocatable derived types.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program defaultmap_implicit_mapper
  implicit none

  type :: payload_t
    integer, allocatable :: arr(:)
  end type payload_t

  type(payload_t), allocatable :: obj
  integer, parameter :: n = 8
  integer :: i
  integer :: scalar
  logical :: ok

  allocate(obj)
  allocate(obj%arr(n))
  obj%arr = 1
  scalar = 2

  !$omp target defaultmap(tofrom: scalar)
    do i = 1, n
      obj%arr(i) = obj%arr(i) + scalar
    end do
    scalar = 7
  !$omp end target

  ok = .true.
  do i = 1, n
    if (obj%arr(i) /= 3) ok = .false.
  end do
  if (scalar /= 7) ok = .false.

  if (ok) then
    print *, "Test passed!"
  else
    print *, "Test failed!"
    print *, obj%arr
    print *, scalar
  end if

  deallocate(obj%arr)
  deallocate(obj)
end program defaultmap_implicit_mapper

! CHECK: Test passed!
