! Offloading test checking a declare mapper whose map clause uses an iterator
! modifier when the iterated array is already present on the device. The
! parent component the mapper synthesizes for the iterated elements spans only
! those elements (not the whole variable), so it is contained in the earlier
! mapping of x%a and must not be rejected by the runtime as an explicit
! extension of it. Reference counts are checked through the matching exits.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program target_declare_mapper_iterator_premapped
  use iso_c_binding, only: c_loc
  use omp_lib, only: omp_get_default_device, omp_target_is_present
  implicit none

  type :: rec_t
    integer :: n
    integer :: a(8)
  end type rec_t

  !$omp declare mapper(m: rec_t :: v) &
  !$omp&  map(iterator(i = 1:4), tofrom: v%a(i))

  type(rec_t), target :: x
  integer :: dev
  logical :: ok

  dev = omp_get_default_device()
  x%n = 4
  x%a = 0

  ! Map the whole array first, then map x through the mapper: its
  ! synthesized parent covers a(1:4), already present within a(1:8).
  !$omp target enter data map(to: x%a)
  ok = (omp_target_is_present(c_loc(x%a(1)), dev) /= 0)
  !$omp target enter data map(mapper(m), to: x)
  ok = ok .and. (omp_target_is_present(c_loc(x%a(1)), dev) /= 0)
  ok = ok .and. (omp_target_is_present(c_loc(x%a(8)), dev) /= 0)

  ! Releasing through the mapper leaves the earlier mapping in place; the
  ! final release removes it.
  !$omp target exit data map(mapper(m), release: x)
  ok = ok .and. (omp_target_is_present(c_loc(x%a(1)), dev) /= 0)
  !$omp target exit data map(release: x%a)
  ok = ok .and. (omp_target_is_present(c_loc(x%a(1)), dev) == 0)

  if (ok) then
    print *, "Test passed!"
  else
    print *, "Test failed!"
  end if
end program target_declare_mapper_iterator_premapped

! CHECK: Test passed!
