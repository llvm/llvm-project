! OpenMP offloading test that checks we do not cause a segfault when mapping
! optional function arguments (present or otherwise). No results requiring
! checking other than that the program compiles and runs to completion with no
! error.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
module foo
    implicit none
    contains
    subroutine test(I,A)
        implicit none
        real(4), optional, intent(inout) :: A(:)
        integer(kind=4), intent(in) :: I

        !$omp target data map(to: A) if (I>0)
        !$omp end target data

        !$omp target enter data map(to:A) if (I>0)

        !$omp target exit data map(from:A) if (I>0)
    end subroutine test
end module foo

program main
    use foo
    implicit none
    real :: array(10)
    call test(0)
    call test(1)
    call test(0, array)
    call test(1, array)
    print *, "PASSED"
end program main

! CHECK: PASSED
