! REQUIRES: flang, amdgpu
! RUN: %libomptarget-compile-fortran-run-and-check-generic

program main
  use omp_lib
  integer :: devices(2), var1
  common var1
  var1 = 10
  print *, "var1 before target = ", var1
  devices(1) = omp_get_device_num()
  !$omp target map(tofrom:devices) map(tofrom:var1)
    var1 = 20
    devices(2) = omp_get_device_num()
  !$omp end target
  print *, "var1 after target = ", var1
  print *, "devices are different? ", (devices(1) /= devices(2))
end program

! CHECK: var1 before target =  10
! CHECK: var1 after target =  20
! CHECK: devices are different? T
