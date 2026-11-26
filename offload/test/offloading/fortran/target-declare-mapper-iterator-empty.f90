! Offloading test checking a declare mapper whose only map clause uses an
! iterator modifier with a runtime range that is empty. Nothing is mapped for
! the variable in that case: the parent component the mapper synthesizes for
! the iterated elements must not be pushed when there are no elements, so a
! `present` modifier on the outer map has nothing to check and the variable
! does not become device-present as a side effect.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program target_declare_mapper_iterator_empty
  use iso_c_binding, only: c_loc
  use omp_lib, only: omp_get_default_device, omp_target_is_present
  implicit none

  type :: rec_t
    integer :: n, a(4)
  end type rec_t

  !$omp declare mapper(m: rec_t :: v) &
  !$omp&  map(iterator(i = 1:v%n), present, to: v%a(i))

  type(rec_t), target :: x
  integer :: dev
  logical :: ok

  dev = omp_get_default_device()
  x%n = 0
  x%a = 1

  ! Empty range: no component is pushed, so `present` must not abort and
  ! nothing of x is mapped.
  !$omp target enter data map(present, mapper(m), to: x)
  ok = (omp_target_is_present(c_loc(x%a(1)), dev) == 0)
  !$omp target exit data map(mapper(m), release: x)

  ! Non-empty range: the same mapper now expands to a(1:4), whose `present`
  ! requires them to be mapped on entry, so map them first. The mapper's
  ! parent spans exactly a(1:4) and so lands on that mapping.
  x%n = 4
  !$omp target enter data map(to: x%a(1:4))
  !$omp target enter data map(present, mapper(m), to: x)
  ok = ok .and. (omp_target_is_present(c_loc(x%a(1)), dev) /= 0)
  ok = ok .and. (omp_target_is_present(c_loc(x%a(4)), dev) /= 0)
  !$omp target exit data map(mapper(m), release: x)
  ok = ok .and. (omp_target_is_present(c_loc(x%a(1)), dev) /= 0)
  !$omp target exit data map(release: x%a(1:4))
  ok = ok .and. (omp_target_is_present(c_loc(x%a(1)), dev) == 0)

  if (ok) then
    print *, "Test passed!"
  else
    print *, "Test failed!"
  end if
end program target_declare_mapper_iterator_empty

! CHECK: Test passed!
