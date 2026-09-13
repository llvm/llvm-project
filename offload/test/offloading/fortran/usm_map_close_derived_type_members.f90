! Offloading test checking map(close, ...) with explicit derived type
! component mappings in unified shared memory mode. When CLOSE is specified
! only on structure components and the base object is not also mapped with
! CLOSE, the compiler drops the CLOSE hint from the component mappings to avoid
! invalid runtime mappings.
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic
! RUN: env HSA_XNACK=1 %libomptarget-run-generic | %fcheck-generic

program main
  !$omp requires unified_shared_memory
  type t
    integer :: x, y
  end type
  type(t) :: member_only = t(1, 2)
  type(t) :: close_parent = t(3, 4)
  type(t) :: nonclose_parent = t(5, 6)

  !$omp target map(close, tofrom: member_only%x, member_only%y)
    member_only%x = 10
    member_only%y = 20
  !$omp end target

  !$omp target map(close, tofrom: close_parent) map(close, tofrom: close_parent%x)
    close_parent%x = 30
    close_parent%y = 40
  !$omp end target

  !$omp target map(tofrom: nonclose_parent) map(close, tofrom: nonclose_parent%x)
    nonclose_parent%x = 50
    nonclose_parent%y = 60
  !$omp end target

  print *, member_only%x, member_only%y
  print *, close_parent%x, close_parent%y
  print *, nonclose_parent%x, nonclose_parent%y
end program

! CHECK: 10 20
! CHECK: 30 40
! CHECK: 50 60
