! Offloading test checking interaction of explicit member mapping of
! non-allocatable members of an allocatable derived type.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: nested_dtype
        real(4) :: i
        real(4) :: j
        integer(4) :: array_i(10)
        integer(4) :: k
    end type nested_dtype

    type :: dtype
        real(4) :: i
        integer, allocatable :: scalar 
        integer(4) :: array_i(10)
        type(nested_dtype) :: nested_dtype
        real(4) :: j
        integer, allocatable :: array_j(:)
        integer(4) :: k
    end type dtype

    type(dtype), allocatable :: alloca_dtype
    allocate(alloca_dtype)

!$omp target map(tofrom: alloca_dtype%nested_dtype%array_i, alloca_dtype%k)
    do i = 1, 10
        alloca_dtype%nested_dtype%array_i(i) = i
    end do
    alloca_dtype%k = 50
!$omp end target

print *, alloca_dtype%k
print *, alloca_dtype%nested_dtype%array_i
end program main

!CHECK: 50
!CHECK: 1 2 3 4 5 6 7 8 9 10
