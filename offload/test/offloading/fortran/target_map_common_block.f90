! Basic offloading test with a target region
! REQUIRES: flang, amdgpu
! RUN: %libomptarget-compile-fortran-run-and-check-generic

! Testing simple variables in common block.
program main
  call check_device
  call commonblock_simple_with_implicit_type_var
  call commonblock_simple_with_integer
  call commonblock_simple_with_real
  call commonblock_simple_to_from
  call set_commonblock_named
  call use_commonblock_named
end program main

!-----

subroutine check_device
  use omp_lib
  integer :: devices(2)
  devices(1) = omp_get_device_num()
  !$omp target map(tofrom:devices)
    devices(2) = omp_get_device_num()
  !$omp end target
  print *, omp_get_num_devices()
  !CHECK: [[ND:[0-9]+]]
  print *, omp_get_default_device()
  !CHECK: [[DD:[0-9]+]]
  !CHECK: devices: [[ND]] [[DD]]
  print *, "devices: ", devices
end subroutine check_device

!-----

subroutine commonblock_simple_with_implicit_type_var
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

! -----

subroutine commonblock_simple_to_from
  use omp_lib
  integer :: var4, tmp
  common var4
  var4 = 10
  tmp = 20
  !$omp target map(to:var4) map(from:tmp)
    tmp = var4
    var4 = 20
  !$omp end target
  print *, "var4 after target = ", var4
  print *, "tmp after target = ", tmp
end subroutine

! CHECK: var4 after target = 10
! CHECK: tmp after target = 10

! -----

subroutine set_commonblock_named
  integer :: var6
  common /my_common_block/ var6
  var6 = 20
end subroutine

subroutine use_commonblock_named
  integer :: var6
  common /my_common_block/ var6
  print *, "var6 before target = ", var6
  !$omp target map(tofrom: var6)
    var6 = 30
  !$omp end target
  print *, "var6 after target = ", var6
end subroutine

! CHECK: var6 before target = 20
! CHECK: var6 after target = 30
