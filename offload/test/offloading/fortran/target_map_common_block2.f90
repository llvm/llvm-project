! REQUIRES: flang, amdgpu
! RUN: %libomptarget-compile-fortran-run-and-check-generic

program main
  use omp_lib
  integer :: tmp, var4
  common var4
  var4 = 24
  tmp = 12
  print *, "var4 before target =", var4
  !$omp target map(tofrom:var4)
    var4 = tmp
  !$omp end target
  print *, "var4 after target =", var4
end program

! CHECK: var4 before target = 24
! CHECK: var4 after target = 12

