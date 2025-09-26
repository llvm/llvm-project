! This test validates that declare mapper for a derived type with an
! allocatable component preserves TO/FROM semantics for the component,
! ensuring the payload is copied back to the host on target exit.

! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program target_declare_mapper_allocatable
  implicit none

  type :: real_t
    real, allocatable :: real_arr(:)
  end type real_t

  ! Map the allocatable array payload via a named mapper.
  !$omp declare mapper (xyz : real_t :: t) map(tofrom: t%real_arr)

  type(real_t) :: r
  integer :: i
  logical :: ok

  allocate(r%real_arr(10))
  r%real_arr = 1.0

  !$omp target map(mapper(xyz), tofrom: r)
    do i = 1, size(r%real_arr)
      r%real_arr(i) = 3.0
    end do
  !$omp end target

  ok = .true.
  do i = 1, size(r%real_arr)
    if (r%real_arr(i) /= 3.0) ok = .false.
  end do
  if (ok) then
    print *, "Test passed!"
  else
    print *, "Test failed!"
    do i = 1, size(r%real_arr)
      print *, r%real_arr(i)
    end do
  end if

  deallocate(r%real_arr)
end program target_declare_mapper_allocatable

! CHECK: Test passed!
