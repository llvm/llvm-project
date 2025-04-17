! Offloading test with a target region mapping a null-ary (no target or
! allocated data) to device, and then setting the target on device before
! printing the changed target on host.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    implicit none
    integer,    pointer :: Set
    integer,    target, allocatable :: Set_Target

    allocate(Set_Target)

    Set_Target = 30

!$omp target map(Set)
    Set => Set_Target
    Set = 45
!$omp end target

    print *, Set_Target
end program main

! CHECK: 45
