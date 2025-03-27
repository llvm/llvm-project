! Offload test that checks an allocatable array can be mapped via member
! mapping alongside a non-allocatable array when both are contained within a
! nested derived type.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: bottom_layer
    real(4) :: i
    integer, allocatable :: scalar_i
    integer(4) :: array_i(10)
    integer, allocatable :: array_k(:)
    integer(4) :: k
    end type bottom_layer

    type :: one_layer
    real(4) :: i
    integer, allocatable :: scalar
    integer(4) :: array_i(10)
    real(4) :: j
    integer, allocatable :: array_j(:)
    integer(4) :: k
    type(bottom_layer) :: nest
    end type one_layer

    type(one_layer) :: one_l

    allocate(one_l%nest%array_k(10))
    allocate(one_l%nest%scalar_i)

    do i = 1, 10
        one_l%nest%array_i(i) = i
    end do

    !$omp target map(tofrom: one_l%nest%array_i, one_l%nest%array_k)
    do i = 1, 10
        one_l%nest%array_k(i) = one_l%nest%array_i(i) + i
    end do
    !$omp end target

    print *, one_l%nest%array_k
    print *, one_l%nest%array_i
end program main

!CHECK: 2 4 6 8 10 12 14 16 18 20
!CHECK: 1 2 3 4 5 6 7 8 9 10

