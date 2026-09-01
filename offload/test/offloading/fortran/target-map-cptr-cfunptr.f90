! Offloading test that verifies a type(c_ptr) and type(c_funptr) nested inside
! a derived type are mapped correctly to the target device, i.e. their
! underlying address values are transferred intact when the enclosing derived
! type is mapped. This exercises the handling of ISO C interoperable types
! nested within derived types across the host/device boundary.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program target_map_cptr_cfunptr
   use iso_c_binding
   implicit none

   type :: iso_holder
      integer        :: id
      type(c_ptr)    :: ptr
      type(c_funptr) :: funptr
   end type iso_holder

   integer, target     :: data
   type(iso_holder)    :: holder
   integer(c_intptr_t) :: host_ptr_val, dev_ptr_val
   integer(c_intptr_t) :: host_fun_val, dev_fun_val

   data = 42

   ! Populate the derived type with the host address values.
   holder%ptr    = c_loc(data)
   holder%funptr = c_funloc(dummy_proc)

   host_ptr_val = transfer(holder%ptr, host_ptr_val)
   host_fun_val = transfer(holder%funptr, host_fun_val)

   dev_ptr_val = 0
   dev_fun_val = 0

   ! Map the whole derived type (containing the nested c_ptr/c_funptr) to the
   ! device and read back their bit values from within the target region.
   !$omp target map(to: holder) &
   !$omp&        map(from: dev_ptr_val, dev_fun_val)
      dev_ptr_val = transfer(holder%ptr, dev_ptr_val)
      dev_fun_val = transfer(holder%funptr, dev_fun_val)
   !$omp end target

   ! CHECK: nested c_ptr mapped correctly
   if (dev_ptr_val == host_ptr_val) then
      print *, "nested c_ptr mapped correctly"
   else
      print *, "nested c_ptr mapping FAILED"
   end if

   ! CHECK: nested c_funptr mapped correctly
   if (dev_fun_val == host_fun_val) then
      print *, "nested c_funptr mapped correctly"
   else
      print *, "nested c_funptr mapping FAILED"
   end if

contains

   subroutine dummy_proc() bind(C)
   end subroutine dummy_proc

end program target_map_cptr_cfunptr
