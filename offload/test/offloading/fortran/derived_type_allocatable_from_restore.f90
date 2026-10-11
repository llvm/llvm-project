! Verify that a real FROM transfer of a derived type containing an attached
! allocatable component restores the complete host descriptor.
!
! REQUIRES: flang, amdgpu, libomptarget-debug
!
! RUN: %libomptarget-compile-fortran-generic && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 | %fcheck-generic

program main
  implicit none

  type :: dtype
    integer, allocatable :: a(:)
    integer :: value
  end type dtype

  type(dtype) :: x

  allocate(x%a(1))
  x%a(1) = 7
  x%value = 1

  !$omp target enter data map(to: x, x%a)

  !$omp target
    x%value = 42
  !$omp end target

  !$omp target exit data map(from: x)

  if (.not. allocated(x%a)) then
    print *, "FAIL: allocatable component is no longer allocated"
    stop 1
  end if

  if (x%a(1) /= 7) then
    print *, "FAIL: allocatable component value changed"
    stop 1
  end if

  if (x%value /= 42) then
    print *, "FAIL: scalar value was not copied back"
    stop 1
  end if

  !$omp target exit data map(delete: x%a)

  deallocate(x%a)

  print *, "PASS"
end program main

! CHECK: Moving {{[0-9]+}} bytes (tgt:{{.*}}) -> (hst:{{.*}})
! CHECK: Restoring host descriptor
! CHECK: PASS
