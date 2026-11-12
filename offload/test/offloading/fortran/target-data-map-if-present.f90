! Offloading test that tests that if(present(a)) compiles and executes without
! causing any compilation errors, primarily a regression test that does not
! yield interesting results.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
module mod
    implicit none
contains
    subroutine routine(a)
        implicit none
        real, dimension(:), optional :: a
        integer :: i
       !$omp target data if(present(a)) map(alloc:a)
            do i = 1, 10
                a(i) = i
            end do
       !$omp end target data
    end subroutine routine
end module mod

program main
    use mod
    real :: a(10)
    call routine(a)
    print *, a
end program main

! CHECK: 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.
