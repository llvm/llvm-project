! Basic offloading test with a target region
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic
program main
   use omp_lib
   implicit none
   integer :: i,j
   integer :: array(10,10), errors = 0
   do i = 1, 10
      do j = 1, 10
         array(j, i) = 0
      end do
   end do

   !$omp target parallel do map(from:array) collapse(2)
   do i = 1, 10
      do j = 1, 10
         array( j, i) = i + j
      end do
    end do
    !$omp end target parallel do

    do i = 1, 10
       do j = 1, 10
          if ( array( j, i) .ne. (i + j) ) then
             errors = errors + 1
          end if
       end do
   end do

   print *,"number of errors: ", errors

end program main

! CHECK:  "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}}
! CHECK:  number of errors: 0

