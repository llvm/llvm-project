! Offloading test checking lowering of arrays with dynamic extents.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program test_openmp_mapper
   implicit none
   integer, parameter :: n = 1024
   type :: mytype
      integer :: data(n)
   end type mytype

   type :: mytype2
      type(mytype) :: my_data
   end type mytype2

   ! Declare custom mappers for the derived type `mytype`
   !$omp declare mapper(my_mapper1 : mytype :: t) map(to: t%data(1 : n))

   ! Declare custom mappers for the derived type `mytype2`
   !$omp declare mapper(my_mapper2 : mytype2 :: t) map(mapper(my_mapper1): t%my_data)

   type(mytype2) :: obj
   integer :: i, sum_host, sum_device

   ! Initialize the host data
   do i = 1, n
      obj%my_data%data(i) = 1
   end do

   ! Compute the sum on the host for verification
   sum_host = sum(obj%my_data%data)

   ! Offload computation to the device using the named mapper `my_mapper2`
   sum_device = 0
   !$omp target map(tofrom: sum_device) map(mapper(my_mapper2) : obj)
   do i = 1, n
      sum_device = sum_device + obj%my_data%data(i)
   end do
   !$omp end target

   ! Check results
   print *, "Sum on host:    ", sum_host
   print *, "Sum on device:  ", sum_device

   if (sum_device == sum_host) then
      print *, "Test passed!"
   else
      print *, "Test failed!"
   end if
 end program test_openmp_mapper

! CHECK:  Test passed!
