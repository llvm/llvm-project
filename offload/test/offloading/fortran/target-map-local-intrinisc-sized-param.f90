! Offloading test checking interaction of an local array
! sized utilising an input parameter and the size intrinsic
! when being mapped to device.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
module mod
    use iso_fortran_env, only: real64
    implicit none
contains
    subroutine test(a)
        implicit none
        integer :: i
        real(kind=real64), dimension(:) :: a
        real(kind=real64), dimension(size(a, 1)) :: b

!$omp target map(tofrom: b)
        do i = 1, 10
            b(i) = i
        end do
!$omp end target

        print *, b
    end subroutine
end module mod

program main
    use mod
    real(kind=real64), allocatable :: a(:)
    allocate(a(10))

    do i = 1, 10
        a(i) = i
    end do

    call test(a)
end program main

!CHECK: 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.
