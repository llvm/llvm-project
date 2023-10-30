! Basic offloading test of a regular array explicitly
! passed within a target region
! REQUIRES: flang, amdgcn-amd-amdhsa
! UNSUPPORTED: nvptx64-nvidia-cuda
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    integer :: x(10) = (/0,0,0,0,0,0,0,0,0,0/)
    integer :: i = 1
    integer :: j = 11

  !$omp target map(tofrom:x, i, j)
     do while (i <= j)
        x(i) = i;
        i = i + 1
    end do
  !$omp end target

   PRINT *, x(:)
end program main
  
! CHECK: 1 2 3 4 5 6 7 8 9 10
