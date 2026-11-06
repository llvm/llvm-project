! Verify that target update to a derived type with an attached pointer component
! restores the target descriptor after the host-to-device update. The update of
! the whole derived type copies the host descriptor bytes to the device, so the
! runtime must put the saved attached target descriptor back before the device
! uses the pointer component again.
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

  ! This H2D update includes the host descriptor for d%p. The runtime should
  ! restore the target descriptor shadow copy after the update.
  !$omp target update to(d)

  got_p = -1
  got_x = -1
  !$omp target map(present, alloc: d) map(from: got_p, got_x)
    got_p = d%p(1)
    got_x = d%x
  !$omp end target

  if (got_p /= 7) then
    print *, "FAIL: target pointer descriptor was not restored"
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

! CHECK: Restoring target descriptor
! CHECK: PASS
