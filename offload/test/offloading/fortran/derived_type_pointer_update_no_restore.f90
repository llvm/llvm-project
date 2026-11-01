! Verify that target update to a non-pointer member of a derived type with an
! attached pointer component does not restore the target descriptor. The update
! range only covers d%x, so it must not overlap the device descriptor slot for
! d%p and the shadow-pointer fixup should be skipped.
!
! REQUIRES: flang, amdgpu, libomptarget-debug
!
! RUN: %libomptarget-compile-fortran-generic && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 | %fcheck-generic

program main
  implicit none

  type :: dtype
    integer, pointer :: p(:) => null()
    integer :: x
  end type dtype

  integer, target :: a(1)
  type(dtype), target :: d
  integer :: got_p, got_x

  a(1) = 7
  d%p => a
  d%x = 1

  !$omp target enter data map(to: a, d, d%p)

  d%x = 42

  ! This H2D update only covers the scalar member d%x. It should not overlap the
  ! attached target descriptor for d%p, so the runtime should not issue a target
  ! descriptor restore.
  !$omp target update to(d%x)

  got_p = -1
  got_x = -1
  !$omp target map(present, alloc: d) map(from: got_p, got_x)
    got_p = d%p(1)
    got_x = d%x
  !$omp end target

  if (got_p /= 7) then
    print *, "FAIL: target pointer descriptor was changed"
    stop 1
  end if

  if (got_x /= 42) then
    print *, "FAIL: scalar was not updated"
    stop 1
  end if

  !$omp target exit data map(release: d%p)
  !$omp target exit data map(delete: d, a)
  nullify(d%p)
  print *, "PASS"
end program main

! CHECK-NOT: Restoring target descriptor
! CHECK: PASS
