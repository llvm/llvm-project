! Regression test for a race at kernel start-up.

! On entry to a target region, thread 0 initializes the team state in LDS
! (HasThreadState = 0, ThreadStates = nullptr) inside __kmpc_target_init and
! then releases the other threads with a barrier. That barrier was emitted with
! `atomic::relaxed`, so the initializing LDS stores are not guaranteed to be
! visible to the worker threads before they proceed. A worker could then read
! uninitialized team state, which showed up as wrong results (or a crash).
!
! The race is timing dependent, so the proven reproducer pattern is repeated
! over a large iteration space to make a bad ordering likely to surface. A
! correct runtime passes every launch deterministically.

! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic
! RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

module control_mod
  implicit none
  ! Not a parameter on purpose: a runtime `if` value is required to triggered
  ! the failure.
  logical :: should_offload = .true.
end module

program target_teams_if_runtime_init
  use control_mod
  implicit none
  integer, parameter :: n = 30000
  integer, parameter :: nlaunch = 50
  integer :: data(n)
  integer :: i, launch, errors

  errors = 0

  do launch = 1, nlaunch
    data = 0

    !$omp target enter data map(alloc: data)

    !$omp target teams distribute parallel do map(present, tofrom: data) &
    !$omp&   if(should_offload)
    do i = 1, n
      data(i) = i
    end do
    !$omp end target teams distribute parallel do

    !$omp target exit data map(from: data)

    do i = 1, n
      if (data(i) /= i) then
        errors = errors + 1
      end if
    end do
  end do

  print *, "number of errors: ", errors

end program target_teams_if_runtime_init

! CHECK: number of errors: 0
