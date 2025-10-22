! This test checks a number of more complex derived type member mapping
! syntaxes utilising an allocatable parent derived type and mixed
! allocatable members.

! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    implicit none

    integer :: i
    integer :: N1, N2

    type :: vertexes
        integer :: test
        integer :: testarray(10)
        integer(4), allocatable :: vertexx(:)
        integer(4), allocatable :: vertexy(:)
    end type vertexes

    type testing_tile_type
        TYPE(vertexes) :: field
    end type testing_tile_type

    type :: dtype
        real(4) :: i
        type(vertexes), allocatable :: vertexes(:)
        TYPE(testing_tile_type), DIMENSION(:), allocatable :: test_tile
        integer(4) :: array_i(10)
        real(4) :: j
        integer, allocatable :: array_j(:)
        integer(4) :: k
    end type dtype

    type(dtype) :: alloca_dtype
    type(dtype), DIMENSION(:), allocatable :: alloca_dtype_arr

    allocate(alloca_dtype%vertexes(4))
    allocate(alloca_dtype%vertexes(1)%vertexx(10))
    allocate(alloca_dtype%vertexes(1)%vertexy(10))
    allocate(alloca_dtype%vertexes(2)%vertexx(10))
    allocate(alloca_dtype%vertexes(2)%vertexy(10))
    allocate(alloca_dtype%vertexes(3)%vertexx(10))
    allocate(alloca_dtype%vertexes(3)%vertexy(10))
    allocate(alloca_dtype%vertexes(4)%vertexx(10))
    allocate(alloca_dtype%vertexes(4)%vertexy(10))
    allocate(alloca_dtype%test_tile(4))
    allocate(alloca_dtype%test_tile(1)%field%vertexx(10))
    allocate(alloca_dtype%test_tile(1)%field%vertexy(10))
    allocate(alloca_dtype%test_tile(2)%field%vertexx(10))
    allocate(alloca_dtype%test_tile(2)%field%vertexy(10))
    allocate(alloca_dtype%test_tile(3)%field%vertexx(10))
    allocate(alloca_dtype%test_tile(3)%field%vertexy(10))
    allocate(alloca_dtype%test_tile(4)%field%vertexx(10))
    allocate(alloca_dtype%test_tile(4)%field%vertexy(10))

    allocate(alloca_dtype_arr(3))

    N1 = 1
    N2 = 2

!$omp target map(tofrom: alloca_dtype%vertexes(N1)%test)
        alloca_dtype%vertexes(N1)%test = 3
!$omp end target

print *, alloca_dtype%vertexes(N1)%test

!$omp target map(tofrom: alloca_dtype%vertexes(N1)%test, alloca_dtype%vertexes(N2)%test)
        alloca_dtype%vertexes(N1)%test = 5
        alloca_dtype%vertexes(N2)%test = 10
!$omp end target

print *, alloca_dtype%vertexes(N1)%test
print *, alloca_dtype%vertexes(N2)%test

!$omp target map(tofrom: alloca_dtype%test_tile(N1)%field%vertexx, &
!$omp                    alloca_dtype%test_tile(N1)%field%vertexy)
    do i = 1, 10
        alloca_dtype%test_tile(N1)%field%vertexx(i) = i + 4
        alloca_dtype%test_tile(N1)%field%vertexy(i) = i + 4
    end do
!$omp end target

print *, alloca_dtype%test_tile(N1)%field%vertexx
print *, alloca_dtype%test_tile(N1)%field%vertexy

!$omp target map(tofrom:  alloca_dtype%test_tile(N1)%field%test, &
!$omp                     alloca_dtype%test_tile(N2)%field%test, &
!$omp                     alloca_dtype%test_tile(N1)%field%vertexy, &
!$omp                     alloca_dtype%test_tile(N2)%field%vertexy)
    alloca_dtype%test_tile(N2)%field%test = 9999
    alloca_dtype%test_tile(N2)%field%vertexy(2) = 9998
    alloca_dtype%test_tile(N1)%field%test = 9997
    alloca_dtype%test_tile(N1)%field%vertexy(2) = 9996
!$omp end target

print *, alloca_dtype%test_tile(N1)%field%test
print *, alloca_dtype%test_tile(N2)%field%test
print *, alloca_dtype%test_tile(N1)%field%vertexy(2)
print *, alloca_dtype%test_tile(N2)%field%vertexy(2)

!$omp target map(tofrom:  alloca_dtype%test_tile(N2)%field%vertexy)
   alloca_dtype%test_tile(N2)%field%vertexy(2) = 2000
!$omp end target

!$omp target map(tofrom: alloca_dtype%vertexes(N1)%vertexx, &
!$omp                    alloca_dtype%vertexes(N1)%vertexy, &
!$omp                    alloca_dtype%vertexes(N2)%vertexx, &
!$omp                    alloca_dtype%vertexes(N2)%vertexy)
    do i = 1, 10
        alloca_dtype%vertexes(N1)%vertexx(i) = i * 2
        alloca_dtype%vertexes(N1)%vertexy(i) = i * 2
        alloca_dtype%vertexes(N2)%vertexx(i) = i * 2
        alloca_dtype%vertexes(N2)%vertexy(i) = i * 2
    end do
!$omp end target

print *, alloca_dtype%vertexes(N1)%vertexx
print *, alloca_dtype%vertexes(N1)%vertexy
print *, alloca_dtype%vertexes(N2)%vertexx
print *, alloca_dtype%vertexes(N2)%vertexy

!$omp target map(tofrom: alloca_dtype%vertexes(N1)%vertexx, &
!$omp                    alloca_dtype%vertexes(N1)%vertexy, &
!$omp                    alloca_dtype%vertexes(4)%vertexy, &
!$omp                    alloca_dtype%vertexes(4)%vertexx, &
!$omp                    alloca_dtype%vertexes(N2)%vertexx, &
!$omp                    alloca_dtype%vertexes(N2)%vertexy)
    do i = 1, 10
        alloca_dtype%vertexes(N1)%vertexx(i) = i * 3
        alloca_dtype%vertexes(N1)%vertexy(i) = i * 3
        alloca_dtype%vertexes(4)%vertexx(i) = i * 3
        alloca_dtype%vertexes(4)%vertexy(i) = i * 3
        alloca_dtype%vertexes(N2)%vertexx(i) = i * 3
        alloca_dtype%vertexes(N2)%vertexy(i) = i * 3
    end do
!$omp end target


    print *, alloca_dtype%vertexes(1)%vertexx
    print *, alloca_dtype%vertexes(1)%vertexy
    print *, alloca_dtype%vertexes(4)%vertexx
    print *, alloca_dtype%vertexes(4)%vertexy
    print *, alloca_dtype%vertexes(2)%vertexx
    print *, alloca_dtype%vertexes(2)%vertexy

!$omp target map(tofrom: alloca_dtype_arr(N2)%array_i)
    do i = 1, 10
        alloca_dtype_arr(N2)%array_i(i) = i + 2
    end do
!$omp end target

print *, alloca_dtype_arr(N2)%array_i

end program main

! CHECK: 3
! CHECK: 5
! CHECK: 10
! CHECK: 5 6 7 8 9 10 11 12 13 14
! CHECK: 5 6 7 8 9 10 11 12 13 14
! CHECK: 9997
! CHECK: 9999
! CHECK: 9996
! CHECK: 9998
! CHECK: 2 4 6 8 10 12 14 16 18 20
! CHECK: 2 4 6 8 10 12 14 16 18 20
! CHECK: 2 4 6 8 10 12 14 16 18 20
! CHECK: 2 4 6 8 10 12 14 16 18 20
! CHECK: 3 6 9 12 15 18 21 24 27 30
! CHECK: 3 6 9 12 15 18 21 24 27 30
! CHECK: 3 6 9 12 15 18 21 24 27 30
! CHECK: 3 6 9 12 15 18 21 24 27 30
! CHECK: 3 6 9 12 15 18 21 24 27 30
! CHECK: 3 6 9 12 15 18 21 24 27 30
! CHECK: 3 4 5 6 7 8 9 10 11 12
