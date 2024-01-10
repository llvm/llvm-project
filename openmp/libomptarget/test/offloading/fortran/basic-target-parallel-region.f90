! Basic offloading test with a target region
! XFAIL: amdgcn-amd-amdhsa
! REQUIRES: flang
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
   use omp_lib
   integer :: x

   !$omp target parallel map(from: x)
         x = omp_get_num_threads()
   !$omp end target parallel
   print *,"parallel = ", (x .ne. 1)

end program main

! CHECK: parallel = T
