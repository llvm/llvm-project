! Basic offloading test with a target region
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic
program main
   use omp_lib
   integer :: x(100)
   integer :: errors = 0
   integer :: i

   !$omp target parallel do map(from: x)
   do i = 1, 100
       x(i) = i
   end do
   !$omp end target parallel do
   do i = 1, 100
       if ( x(i) .ne. i ) then
           errors = errors + 1
       end if
   end do

   print *,"number of errors: ", errors

end program main

! CHECK:  "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}}
! CHECK:  number of errors: 0
