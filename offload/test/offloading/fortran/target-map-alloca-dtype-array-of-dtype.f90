! Offloading test checking interaction of explicit member mapping of an array
! of derived types contained within an allocatable derived type
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
        integer(4) :: array_i(10)
        real(4) :: j
        type(nested_dtype) :: array_dtype(10)
        integer(4) :: k
    end type dtype

    type(dtype), allocatable :: dtyped
    allocate(dtyped)

!$omp target map(tofrom: dtyped%array_dtype)
    do i = 1, 10
        dtyped%array_dtype(i)%k = i
    end do
!$omp end target

print *, dtyped%array_dtype%k

end program main

!CHECK: 1 2 3 4 5 6 7 8 9 10
