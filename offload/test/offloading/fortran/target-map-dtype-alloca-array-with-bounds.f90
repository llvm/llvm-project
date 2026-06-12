! Offload test that checks an allocatable array can be mapped with a specified
! 1-D bounds when contained within a derived type and mapped via member mapping
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: top_layer
    real(4) :: i
    integer, allocatable :: scalar
    integer(4) :: array_i(10)
    real(4) :: j
    integer, allocatable :: array_j(:)
    integer(4) :: k
    end type top_layer

    type(top_layer) :: one_l

    allocate(one_l%array_j(10))

!$omp target map(tofrom: one_l%array_j(2:6))
    do index = 1, 10
        one_l%array_j(index) = index
    end do
!$omp end target

    print *, one_l%array_j(2:6)
end program main

!CHECK: 2 3 4 5 6
