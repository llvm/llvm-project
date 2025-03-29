! Basic offloading test with a target region
! REQUIRES: flang, amdgpu

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
