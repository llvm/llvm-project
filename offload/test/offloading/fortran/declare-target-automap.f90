!Offloading test for AUTOMAP modifier in declare target enter
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program automap_program
   use iso_c_binding, only: c_loc
   use omp_lib, only: omp_get_default_device, omp_target_is_present
   integer, parameter :: N = 10
   integer :: i
   integer, allocatable, target :: automap_array(:)
   !$omp declare target enter(automap:automap_array)

   ! false since the storage is not present even though the descriptor is present
   write (*, *) omp_target_is_present(c_loc(automap_array), omp_get_default_device())
   ! CHECK: 0

   allocate (automap_array(N))
   ! true since the storage should be allocated and reference count incremented by the allocate
   write (*, *) omp_target_is_present(c_loc(automap_array), omp_get_default_device())
   ! CHECK: 1

   ! since storage is present this should not be a runtime error
   !$omp target teams loop
   do i = 1, N
      automap_array(i) = i
   end do

   !$omp target update from(automap_array)
   write (*, *) automap_array
   ! CHECK: 1 2 3 4 5 6 7 8 9 10

   deallocate (automap_array)

   ! automap_array should have it's storage unmapped on device here
   write (*, *) omp_target_is_present(c_loc(automap_array), omp_get_default_device())
   ! CHECK: 0
end program
