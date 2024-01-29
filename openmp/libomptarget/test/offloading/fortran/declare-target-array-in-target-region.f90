! Offloading test with a target region mapping a declare target
! Fortran array writing some values to it and checking the host
! correctly receives the updates made on the device.
! REQUIRES: flang, amdgcn-amd-amdhsa, nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
module test_0
    implicit none
    INTEGER :: sp(10) = (/0,0,0,0,0,0,0,0,0,0/)
    !$omp declare target link(sp)
end module test_0

program main
    use test_0
    integer :: i = 1
    integer :: j = 11
!$omp target map(tofrom:sp, i, j)
    do while (i <= j)
        sp(i) = i;
        i = i + 1
    end do
!$omp end target

PRINT *, sp(:)

end program

! CHECK: 1 2 3 4 5 6 7 8 9 10
