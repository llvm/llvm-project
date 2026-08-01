! Show that the ALWAYS map-type modifier on the outer map clause should be
! propagated to the entries pushed by a user-defined mapper. The test uses no
! target region at all: the device data is inspected with omp_get_mapped_ptr()
! and omp_target_memcpy(), so what is checked is purely the data-motion done by
! `target enter data`.
!
! The mapper transfers s%y. We pre-map s%y so that it already has a device copy
! with a nonzero reference count, and set that copy to a known value. Without
! ALWAYS, the `to` of the second `enter data` is suppressed (a present,
! ref-counted entry is not copied), so the host's 111 does not reach the device.
! ALWAYS must force the copy -- but only if ALWAYS is propagated from the outer
! clause to the mapper's s%y entry.
!
! FIXME: ALWAYS is not propagated to the mapper's entries yet, so the transfer
! is currently suppressed and the device copy still reads as 0 (its pre-set
! value). Once ALWAYS is propagated:
!   EXPECTED: device s%y = 111

! REQUIRES: flang

! RUN: %libomptarget-compile-fortran-generic
! RUN: %libomptarget-run-generic | %fcheck-generic

program main
  use omp_lib
  use iso_c_binding
  implicit none

  type :: s_type
    integer(c_int) :: x
    integer(c_int) :: y
    integer(c_int) :: z
  end type s_type

  !$omp declare mapper(s_type :: s) map(tofrom: s%y)

  type(s_type), target :: s
  integer(c_int), target :: dev_y_val
  integer :: dev, host, rc
  type(c_ptr) :: dev_y

  dev = omp_get_default_device()
  host = omp_get_initial_device()

  s%y = 0

  !$omp target enter data map(alloc: s%y)

  ! The device copy of s%y is uninitialized after `alloc`; set it to 0 so that
  ! a missing transfer below is distinguishable from garbage.
  dev_y = omp_get_mapped_ptr(c_loc(s%y), dev)
  rc = omp_target_memcpy(dev_y, c_loc(s%y), int(c_sizeof(s%y), c_size_t), &
                         0_c_size_t, 0_c_size_t, dev, host)

  s%y = 111

  !$omp target enter data map(always, to: s)

  ! ALWAYS should force s%y to the device even though it is already mapped
  ! (ref count > 0), but the modifier is not propagated yet, so the transfer
  ! does not happen.
  dev_y_val = -1
  rc = omp_target_memcpy(c_loc(dev_y_val), dev_y, &
                         int(c_sizeof(dev_y_val), c_size_t), &
                         0_c_size_t, 0_c_size_t, host, dev)

  print *, "device s%y =", dev_y_val
  ! CHECK: device s%y = 0

  !$omp target exit data map(delete: s%y)
end program main
