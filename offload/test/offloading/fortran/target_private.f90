! Basic offloading test with a target region
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program target_update
    implicit none
    integer :: x(1)
    integer :: y(1)

    x(1) = 42

!$omp target private(x) map(tofrom: y)
    x(1) = 84
    y(1) = x(1)
!$omp end target

    print *, "x =", x(1)
    print *, "y =", y(1)

end program target_update

! CHECK: x = 42
! CHECK: y = 84
