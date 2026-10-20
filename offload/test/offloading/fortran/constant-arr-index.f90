! Basic offloading test with a target region that checks constant indexing on
! device correctly works (regression test for prior bug).
! REQUIRES: flang, amdgpu

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
