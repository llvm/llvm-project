! Test `declare target to` interaction with an allocatable with a non-default
! range
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
module test_0
    real(4), allocatable          ::   zero_off(:)
    !$omp declare target(zero_off)
end module test_0

program main
    use test_0
    implicit none

    allocate(zero_off(0:10))

    zero_off(0) = 30.0
    zero_off(1) = 40.0
    zero_off(10) = 25.0

    !$omp target map(tofrom: zero_off)
        zero_off(0) = zero_off(1)
    !$omp end target

    print *, zero_off(0)
    print *, zero_off(1)
end program

! CHECK: 40.
! CHECK: 40.
