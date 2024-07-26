! Offloading test checking interaction of
! mapping all the members of a common block
! to a target region
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    implicit none
    common /var_common/ var1, var2, var3
    integer :: var1, var2, var3

    call modify_1

  !$omp target map(tofrom: var1, var2, var3)
    var3 = var3 * 10
    var2 = var2 * 10
    var1 = var1 * 10
  !$omp end target

  call modify_2

  print *, var1
  print *, var2
  print *, var3
end program

subroutine modify_1
  common /var_common/ var1, var2, var3
  integer :: var1, var2, var3
!$omp target map(tofrom: var2, var1, var3)
  var3 = var3 + 40
  var2 = var2 + 20
  var1 = var1 + 30
!$omp end target
end

subroutine modify_2
  common /var_common/ var1, var2, var3
  integer :: var1, var2, var3
!$omp target map(tofrom: var2, var3, var1)
  var3 = var3 + 20
  var1 = var1 + 10
  var2 = var2 + 15
!$omp end target
end

!CHECK: 310
!CHECK: 215
!CHECK: 420
