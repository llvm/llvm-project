! Offloading test checking interaction of
! mapping a full common block in a target
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
    common /var_common/ var1, var2
    integer :: var1, var2

    call modify_1

  !$omp target map(tofrom: /var_common/)
      var1 = var1 + 20
      var2 = var2 + 50
  !$omp end target

    call modify_2

    print *, var1
    print *, var2
end program

subroutine modify_1
  common /var_common/ var1, var2
  integer :: var1, var2
!$omp target map(tofrom: /var_common/)
  var1 = var1 + 20
  var2 = var2 + 30
!$omp end target
end

subroutine modify_2
  common /var_common/ var1, var2
  integer :: var1, var2
!$omp target map(tofrom: /var_common/)
  var1 = var1 * 10
  var2 = var2 * 10
!$omp end target
end

!CHECK: 400
!CHECK: 800
