! Basic offloading test with a target region
! that checks constant indexing on device
! correctly works (regression test for prior
! bug).
! REQUIRES: flang
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO
! UNSUPPORTED: aarch64-unknown-linux-gnu
! UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
! UNSUPPORTED: x86_64-pc-linux-gnu
! UNSUPPORTED: x86_64-pc-linux-gnu-LTO

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    INTEGER :: sp(10) = (/0,0,0,0,0,0,0,0,0,0/)

  !$omp target map(tofrom:sp)
     sp(1) = 20
     sp(5) = 10
  !$omp end target

   print *, sp(1)
   print *, sp(5)
end program

! CHECK: 20
! CHECK: 10
