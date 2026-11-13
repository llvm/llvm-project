! Offloading test checking a declare mapper that maps both the whole variable
! and, via an iterator modifier, individual elements of a component, used with
! TARGET ENTER DATA / TARGET EXIT DATA. The iterated elements must be grouped
! under the existing whole-variable entry (MEMBER_OF) so that the exit-data
! transfer brings back the values written on the device.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program target_declare_mapper_iterator_enter_exit
  implicit none
  integer, parameter :: n = 6

  type :: vec_t
    integer :: tag
    integer :: a(n)
  end type vec_t

  !$omp declare mapper(m: vec_t :: v) map(v) map(iterator(i = 1:n): v%a(i))

  type(vec_t) :: x
  integer :: i
  logical :: ok

  x%tag = 7
  x%a = 0

  !$omp target enter data map(mapper(m), to: x)

  !$omp target
    x%tag = x%tag + 1
    do i = 1, n
      x%a(i) = x%tag * i
    end do
  !$omp end target

  !$omp target exit data map(mapper(m), from: x)

  ok = (x%tag == 8)
  do i = 1, n
    if (x%a(i) /= 8 * i) ok = .false.
  end do

  if (ok) then
    print *, "Test passed!"
  else
    print *, "Test failed!"
    print *, x%tag, x%a
  end if
end program target_declare_mapper_iterator_enter_exit

! CHECK: Test passed!
