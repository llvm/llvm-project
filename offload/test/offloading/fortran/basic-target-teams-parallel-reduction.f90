! Basic offloading test with a target region
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic
program main
   use omp_lib
   integer :: error = 0
   integer :: i
   integer :: sum = 0

   !$omp target teams distribute parallel do reduction(+:sum)
   do i = 1, 1000
       sum = sum + i
   end do
   !$omp end target teams distribute parallel do

   if (sum /= 500500) then
     error = 1
  endif

  print *,"number of errors: ", error

end program main

! CHECK:  "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}}
! CHECK:  number of errors: 0
