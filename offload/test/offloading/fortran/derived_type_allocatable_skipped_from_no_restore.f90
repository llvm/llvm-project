! Verify that a skipped FROM transfer of a derived type containing an attached
! allocatable component does not restore the host descriptor. If no D2H copy
! overwrote the host descriptor bytes, restoring from shadow pointer info would
! be unnecessary and can clobber valid host-side descriptor state.
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

  ! The target region increments the refcount for x and then decrements it back
  ! to the still-live enter-data mapping. The FROM transfer for x is skipped
  ! because x is not the last user. Since no host bytes are overwritten by a D2H
  ! copy, target-data-end post-processing must not restore the host descriptor.
  !$omp target map(tofrom: x)
    x%value = 42
  !$omp end target

  if (.not. allocated(x%a)) then
    print *, "FAIL: allocatable component is no longer allocated"
    stop 1
  end if

  if (x%a(1) /= 7) then
    print *, "FAIL: allocatable component value changed"
    stop 1
  end if

  if (x%value /= 1) then
    print *, "FAIL: scalar value unexpectedly copied back"
    stop 1
  end if

  !$omp target exit data map(delete: x)
  !$omp target exit data map(delete: x%a)

  deallocate(x%a)
  print *, "PASS"
end program main

! CHECK: Skipping FROM map transfer
! CHECK-NOT: Restoring host descriptor
! CHECK: PASS
