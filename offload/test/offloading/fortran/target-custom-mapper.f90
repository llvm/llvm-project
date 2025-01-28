! Offloading test checking lowering of arrays with dynamic extents.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic

program test_openmp_mapper
  implicit none
  integer, parameter :: n = 1024
  type :: mytype
     integer :: data(n)
  end type mytype

  ! Declare a custom mapper for the derived type `mytype` with the name `my_mapper`
  !$omp declare mapper(my_mapper : mytype :: t) map(to: t%data)

  type(mytype) :: obj
  integer :: i, sum_host, sum_device

  ! Initialize the host data
  do i = 1, n
     obj%data(i) = 1
  end do

  ! Compute the sum on the host for verification
  sum_host = sum(obj%data)

  ! Offload computation to the device using the named mapper `my_mapper`
  sum_device = 0
  !$omp target map(tofrom: sum_device) map(mapper(my_mapper) : obj)
  do i = 1, n
     sum_device = sum_device + obj%data(i)
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
