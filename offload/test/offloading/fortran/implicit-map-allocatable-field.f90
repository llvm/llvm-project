! Verify that a derived type with an allocatable component, captured implicitly
! by a target region, is correctly mapped to and from the device.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program p
  implicit none
  type t
    integer, allocatable :: a(:)
  end type
  type(t) :: x

  allocate(x%a(1))
  x%a = 0

  !$omp target
    x%a(1) = 42
  !$omp end target

  if (x%a(1) /= 42) then
    print *, "======= Test Failed! ======="
    stop 1
  end if

  print *, "======= Test Passed! ======="

  deallocate(x%a)
end program

! CHECK: ======= Test Passed! =======
