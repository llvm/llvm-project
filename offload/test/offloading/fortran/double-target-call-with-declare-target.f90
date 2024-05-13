! Offloading test with two target regions mapping the same
! declare target Fortran array and writing some values to 
! it before checking the host correctly receives the 
! correct updates made on the device.
! REQUIRES: flang
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
module test_0
    implicit none
    integer :: sp(10) = (/0,0,0,0,0,0,0,0,0,0/)
    !$omp declare target link(sp)
end module test_0

program main
    use test_0
    integer :: i = 1
    integer :: j = 11

!$omp target map(tofrom:sp) map(to: i, j)
    do while (i <= j)
        sp(i) = i;
        i = i + 1
    end do
!$omp end target

!$omp target map(tofrom:sp) map(to: i, j)
    do while (i <= j)
        sp(i) = sp(i) + i;
        i = i + 1
    end do
!$omp end target
    
print *, sp(:)

end program

! CHECK: 2 4 6 8 10 12 14 16 18 20
