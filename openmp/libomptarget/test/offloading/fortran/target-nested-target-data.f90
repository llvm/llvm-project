! Offloading test for target nested inside
! a target data region
! REQUIRES: flang
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
   integer :: A(10), B(10), C(10)

   do I = 1, 10
      A(I) = 1
      B(I) = 2
   end do
   !$omp target data map(to: A, B) map(alloc: C)
   !$omp target map(from: C)
   do I = 1, 10
      C(I) = A(I) + B(I) ! assigns 3, A:1 + B:2
   end do
   !$omp end target
   !$omp target update from(C) ! updates C device -> host
   !$omp end target data

   print *, C ! should be all 3's

end program

! CHECK: 3 3 3 3 3 3 3 3 3 3
