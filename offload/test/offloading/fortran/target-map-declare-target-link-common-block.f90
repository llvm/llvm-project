! Offloading test checking interaction of
! mapping a declare target link common
! block with device_type any to a target
! region
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-unknown-linux-gnu
! UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    implicit none
    common /var_common/ var1, var2, var3
    integer :: var1, var2, var3
    !$omp declare target link(/var_common/)

    call modify_1

  !$omp target map(tofrom: var2)
    var2 = var2 + var3
  !$omp end target

  call modify_2
  call modify_3

  print *, var1
  print *, var2
  print *, var3
end program

subroutine modify_1
  common /var_common/ var1, var2, var3
  integer :: var1, var2, var3

!$omp target map(tofrom: /var_common/)
  var1 = 10
  var2 = 20
  var3 = 30
!$omp end target

end

subroutine modify_2
  common /var_common/ var1, var2, var3
  integer :: var1, var2, var3
  integer :: copy

!$omp target map(tofrom: copy)
  copy =  var2 + var3
!$omp end target

  print *, copy
end

subroutine modify_3
  common /var_common/ var1, var2, var3
  integer :: var1, var2, var3

!$omp target map(tofrom: /var_common/)
  var1 = var1 + var1
  var2 = var2 + var2
  var3 = var3 + var3
!$omp end target
end

!CHECK: 80
!CHECK: 20
!CHECK: 100
!CHECK: 60