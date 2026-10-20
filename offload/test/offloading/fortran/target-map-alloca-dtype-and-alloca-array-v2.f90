! Offload test that checks an allocatable derived type can be mapped alongside
! one of its own allocatable components without disrupting either mapping,
! different from original as the argument ordering is reversed (similar to C++
! mapping of a struct and a pointer, in concept at least).
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: dtype
        real(4) :: i
        integer, allocatable :: scalar
        integer(4) :: array_i(10)
        real(4) :: j
        integer, allocatable :: array_j(:)
        integer(4) :: k
    end type dtype

    type(dtype), allocatable :: alloca_dtype
    allocate(alloca_dtype)
    allocate(alloca_dtype%array_j(10))

!$omp target map(tofrom: alloca_dtype%array_j, alloca_dtype)
    do i = 1, 10
        alloca_dtype%array_j(i) = i
    end do
    alloca_dtype%k = 50
!$omp end target

print *, alloca_dtype%array_j
print *, alloca_dtype%k
end program main

!CHECK: 1 2 3 4 5 6 7 8 9 10
!CHECK: 50
