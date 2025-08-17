! Offloading test checking interaction of an explicit member map of an
! allocatable array within a nested allocatable derived type with specified
! bounds
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: bottom_layer
    real(4) :: i
    integer(4) :: array_i(10)
    integer, allocatable :: array_k(:)
    integer(4) :: k
    end type bottom_layer

    type :: top_layer
    real(4) :: i
    integer, allocatable :: scalar
    integer(4) :: array_i(10)
    real(4) :: j
    integer, allocatable :: array_j(:)
    integer(4) :: k
    type(bottom_layer), allocatable :: nest
    end type top_layer

    type(top_layer), allocatable :: one_l
    allocate(one_l)
    allocate(one_l%nest)
    allocate(one_l%nest%array_k(10))

    do index = 1, 10
        one_l%nest%array_k(index) = 0
    end do

!$omp target map(tofrom: one_l%nest%array_k(2:6))
    do index = 2, 6
        one_l%nest%array_k(index) = index
    end do
!$omp end target

    print *, one_l%nest%array_k
end program main

!CHECK: 0 2 3 4 5 6 0 0 0 0
