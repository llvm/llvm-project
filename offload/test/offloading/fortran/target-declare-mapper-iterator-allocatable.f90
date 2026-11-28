! Offloading test checking a declare mapper whose iterator modifier maps
! elements of an ALLOCATABLE array component. The elements live in storage
! reached through the component's descriptor, so the mapper maps the
! descriptor, a storage block spanning the elements, and attaches the device
! descriptor to it; a kernel then reaches the elements through the descriptor.
! Elements outside the iterator's range stay unmapped, and an empty range maps
! no elements at all.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program target_declare_mapper_iterator_allocatable
  use iso_c_binding, only: c_loc
  use omp_lib, only: omp_get_default_device, omp_target_is_present
  implicit none

  type :: vec_t
    integer :: n
    integer, allocatable :: a(:)
  end type vec_t

  !$omp declare mapper(m: vec_t :: v) map(to: v%n) &
  !$omp&  map(iterator(i = 1:v%n), tofrom: v%a(i))

  type(vec_t), target :: x
  integer :: i, dev
  logical :: ok

  dev = omp_get_default_device()
  allocate(x%a(8))
  x%a = 0

  ! Range a(1:4) of the 8 elements: the kernel reads and writes them through
  ! the (attached) device descriptor.
  x%n = 4
  !$omp target map(mapper(m): x)
    do i = 1, x%n
      x%a(i) = 10 * i + size(x%a)
    end do
  !$omp end target

  ok = .true.
  do i = 1, 4
    if (x%a(i) /= 10 * i + 8) ok = .false.
  end do
  do i = 5, 8
    if (x%a(i) /= 0) ok = .false.
  end do

  ! enter/exit data: the mapped range is device-present until released, the
  ! rest of the array never is.
  !$omp target enter data map(mapper(m), to: x)
  ok = ok .and. (omp_target_is_present(c_loc(x%a(1)), dev) /= 0)
  ok = ok .and. (omp_target_is_present(c_loc(x%a(4)), dev) /= 0)
  ok = ok .and. (omp_target_is_present(c_loc(x%a(5)), dev) == 0)
  !$omp target exit data map(mapper(m), release: x)
  ok = ok .and. (omp_target_is_present(c_loc(x%a(1)), dev) == 0)

  ! Empty range: no element is mapped.
  x%n = 0
  !$omp target enter data map(mapper(m), to: x)
  ok = ok .and. (omp_target_is_present(c_loc(x%a(1)), dev) == 0)
  !$omp target exit data map(mapper(m), release: x)

  if (ok) then
    print *, "Test passed!"
  else
    print *, "Test failed!"
    print *, x%a
  end if
end program target_declare_mapper_iterator_allocatable

! CHECK: Test passed!
