! Basic offloading test with a target region
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic

! Testing simple variables in common block.
program main
  call check_device
  call commonblock_simple_with_implicit_type
  call commonblock_simple_with_integer
  call commonblock_simple_with_real
end program main

!-----

subroutine check_device
  use omp_lib
  integer :: devices(2)
  devices(1) = omp_get_device_num()
  !$omp target map(tofrom:devices)
    devices(2) = omp_get_device_num()
  !$omp end target
  print *, "devices: ", devices
end subroutine check_device

!CHECK: devices: 1 0

!-----

subroutine commonblock_simple_with_implicit_type
  use omp_lib
  common var1
  var1 = 10
  print *, "var1 before target = ", var1
  !$omp target map(tofrom:var1)
    var1 = 20
  !$omp end target
  print *, "var1 after target = ", var1
end subroutine

! CHECK: var1 before target = 10
! CHECK: var1 after target = 20

! -----

subroutine commonblock_simple_with_integer
  use omp_lib
  integer :: var2
  common var2
  var2 = 10
  print *, "var2 before target = ", var2
  !$omp target map(tofrom:var2)
    var2 = 20
  !$omp end target
  print *, "var2 after target = ", var2
end subroutine

! CHECK: var2 before target = 10
! CHECK: var2 after target = 20

! -----

subroutine commonblock_simple_with_real
  use omp_lib
  real :: var3
  common var3
  var3 = 12.5
  print *, "var3 before target = ", var3
  !$omp target map(tofrom:var3)
    var3 = 14.5
  !$omp end target
  print *, "var3 after target = ", var3
end subroutine

! CHECK: var3 before target = 12.5
! CHECK: var3 after target = 14.5
