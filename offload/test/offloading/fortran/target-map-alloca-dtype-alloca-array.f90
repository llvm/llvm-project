! Offload test that checks an allocatable array within an allocatable derived
! type can be mapped explicitly using member mapping.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    type :: dtype
        real(4) :: i
        integer(4) :: array_i(10)
        real(4) :: j
        integer, allocatable :: array_j(:)
        integer(4) :: k
    end type dtype

    type(dtype), allocatable :: alloca_dtype
    allocate(alloca_dtype)
    allocate(alloca_dtype%array_j(10))

!$omp target map(tofrom: alloca_dtype%array_j)
    do i = 1, 10
        alloca_dtype%array_j(i) = i
    end do
!$omp end target

print *, alloca_dtype%array_j

end program main

!CHECK: 1 2 3 4 5 6 7 8 9 10
