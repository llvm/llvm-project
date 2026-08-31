! Verifies that `do concurrent` correctly lowers map for basic nested derived
! types, correctly mapping the required components of the derived type.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic -fdo-concurrent-to-openmp=device
! RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic
program main
   implicit none

   type :: alloc_buffer
     integer :: i
     real, allocatable :: data(:)
   end type alloc_buffer

   type :: buffer
     integer :: i
     real :: data(8)
   end type buffer

   type :: array_dt
      type(buffer) :: buf
   end type array_dt

   type :: alloc_array_dt
      type(alloc_buffer) :: buf
   end type alloc_array_dt

   integer, parameter :: n = 8
   integer :: i
   type(alloc_array_dt) :: aad
   type(array_dt) :: ad

   allocate(aad%buf%data(n), source=0.0)

   do concurrent(i=1:n)
      aad%buf%data(i) = real(i)
   end do

   do concurrent(i=1:n)
      ad%buf%data(i) = real(i)
   end do

   print *, sum(ad%buf%data)
   print *, sum(aad%buf%data)

   deallocate(aad%buf%data)
end program main

! CHECK: 36.
! CHECK: 36.
