! Offload test that checks an allocatable derived type can be mapped and at the
! least non-allocatable components can be accessed.
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

!$omp target map(tofrom: alloca_dtype)
    do i = 1, 10
        alloca_dtype%array_i(i) = i
    end do
    alloca_dtype%k = 50
!$omp end target

print *, alloca_dtype%k
print *, alloca_dtype%array_i
end program main

!CHECK: 50
!CHECK: 1 2 3 4 5 6 7 8 9 10
