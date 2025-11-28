! Basic offloading test checking the interaction of an overlapping
! member map.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    implicit none
    integer :: i

    type dtype2
        integer :: int
        real :: float
    end type dtype2

    type dtype1
        character (LEN=30) :: characters
        type(dtype2) :: internal_dtype2
    end type dtype1

    type dtype
        integer :: elements(10)
        type(dtype1) :: internal_dtype
        integer :: value
    end type dtype

    type (dtype) :: single_dtype

    do i = 1, 10
      single_dtype%elements(i) = 0
    end do

  !$omp target map(tofrom: single_dtype%internal_dtype, single_dtype%internal_dtype%internal_dtype2%int)
    single_dtype%internal_dtype%internal_dtype2%int = 123
    single_dtype%internal_dtype%characters(1:1) = "Z"
  !$omp end target

  !$omp target map(to: single_dtype) map(tofrom: single_dtype%internal_dtype%internal_dtype2, single_dtype%value)
    single_dtype%value = 20
    do i = 1, 10
      single_dtype%elements(i) = i
    end do
    single_dtype%internal_dtype%internal_dtype2%float = 32.0
  !$omp end target

  print *, single_dtype%value
  print *, single_dtype%internal_dtype%internal_dtype2%float
  print *, single_dtype%elements
  print *, single_dtype%internal_dtype%internal_dtype2%int
  print *, single_dtype%internal_dtype%characters(1:1)
end program main

! CHECK: 20
! CHECK: 32.
! CHECK: 0 0 0 0 0 0 0 0 0 0
! CHECK: 123
! CHECK: Z
