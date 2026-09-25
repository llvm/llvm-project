! Verifies that `do concurrent` correctly lowers map for basic nested derived
! types, correctly mapping the required components of the derived type.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic -fdo-concurrent-to-openmp=device
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic
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
      type(alloc_buffer) :: buf(2)
   end type alloc_array_dt

   integer, parameter :: n = 8
   integer :: i
   type(alloc_array_dt) :: aad
   type(array_dt) :: ad

   allocate(aad%buf(1)%data(n), source=0.0)
   allocate(aad%buf(2)%data(n), source=0.0)

   do concurrent(i=1:n)
      aad%buf(1)%data(i) = real(i)
      aad%buf(2)%data(i) = real(i * 2)
   end do

   do concurrent(i=1:n)
      ad%buf%data(i) = real(i)
   end do

   if (sum(ad%buf%data) == 36.0 .and. &
       sum(aad%buf(1)%data) == 36.0 .and. &
       sum(aad%buf(2)%data) == 72.0) then
      print *, "PASS"
   else
      print *, "FAIL"
   end if

   deallocate(aad%buf(1)%data)
   deallocate(aad%buf(2)%data)
end program main

! CHECK:  PluginInterface device {{[0-9]+}} info: Launching kernel {{.*}}
! CHECK:  PluginInterface device {{[0-9]+}} info: Launching kernel {{.*}}
! CHECK:  PASS
