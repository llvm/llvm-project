! REQUIRES: flang, flang-rt, libc, amdgpu
! UNSUPPORTED: apu
! RUN: %libomptarget-compile-fortran-run-and-check-generic

! Empty dimensions must remain empty when contiguous dimensions are merged.
program target_update_empty
  implicit none
  integer, allocatable :: a(:,:)
  integer :: storage(4,3), actual(4,3), n, lo, hi
  integer, parameter :: host_sentinel = 7777, device_sentinel = -9999

  n = 0
  allocate(a(n,3))
!$omp target enter data map(to:a)
!$omp target update to(present:a)
!$omp target update from(present:a)
!$omp target update to(present:a(:,2:))
!$omp target update from(present:a(:,2:))
!$omp target exit data map(delete:a)
  deallocate(a)

  allocate(a(4,n))
!$omp target enter data map(to:a)
!$omp target update to(present:a)
!$omp target update from(present:a)
!$omp target update to(present:a(2:3,:))
!$omp target update from(present:a(2:3,:))
!$omp target exit data map(delete:a)
  deallocate(a)

  ! Empty sections of nonempty arrays must preserve every surrounding element.
  storage = host_sentinel
!$omp target enter data map(alloc:storage)
!$omp target map(present,alloc:storage)
  storage = device_sentinel
!$omp end target
  lo = 3
  hi = 2
!$omp target update to(present:storage(lo:hi,:))
!$omp target update to(present:storage(:,lo:hi))
!$omp target map(present,alloc:storage) map(from:actual)
  actual = storage
  storage = -42
!$omp end target
  if (any(actual /= device_sentinel)) stop 1
!$omp target update from(present:storage(lo:hi,:))
!$omp target update from(present:storage(:,lo:hi))
  if (any(storage /= host_sentinel)) stop 2
!$omp target exit data map(delete:storage)
  print *, 'PASS'
end program

! CHECK: PASS
