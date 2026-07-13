! Offloading test that verifies certain type of character string arrays
! map to and from device without problem.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
  implicit none
  type char_t
    CHARACTER(LEN=16), dimension(10,10) :: char_arr
  end type char_t
  type(char_t) :: dtype_char

!$omp target enter data map(alloc:dtype_char%char_arr)

!$omp target
    dtype_char%char_arr(2,2) = 'c'
!$omp end target

!$omp target update from(dtype_char%char_arr)


 print *, dtype_char%char_arr(2,2)
end program

!CHECK: c
