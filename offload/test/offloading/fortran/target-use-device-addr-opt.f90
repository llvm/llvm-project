! REQUIRES: flang
! REQUIRES: gpu, amdgpu

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=8  %libomptarget-run-generic 2>&1 | %fcheck-generic
MODULE foo
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: bar_device_addr

CONTAINS

  SUBROUTINE bar_device_addr(x)
    INTEGER, TARGET, INTENT(IN)    :: x(:)
    !$omp target data use_device_addr (x)
    !$omp end target data
  END SUBROUTINE

END MODULE foo

PROGRAM test_ptr
  USE, intrinsic :: iso_fortran_env, only: error_unit
  USE foo
  IMPLICIT NONE

  INTEGER, ALLOCATABLE, TARGET :: x(:)
  ALLOCATE(x(10))
  !$omp target enter data map(to: x)
    CALL bar_device_addr(x)
  !$omp target exit data map(from: x)
  DEALLOCATE(x)
  write(error_unit, *) 'Success'
END PROGRAM test_ptr

! CHECK: Creating new map entry
! CHECK: Creating new map entry
! CHECK-NOT: Creating new map entry
! CHECK: Removing map entry
! CHECK: Removing map entry
! CHECK-NOT: Removing map entry
! CHECK: Success
