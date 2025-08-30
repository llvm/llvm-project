! Check if the first OpenMP GPU kernel is promoted to no-loop mode.
! The second cannot be promoted due to the limit on the number of teams.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic -O3  -fopenmp-assume-threads-oversubscription -fopenmp-assume-teams-oversubscription
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic
program main
   use omp_lib
   implicit none
   integer :: i
   integer :: array(1024), errors = 0
   array = 1

   !$omp target teams distribute parallel do
   do i = 1, 1024
      array(i) = i
    end do

   do i = 1, 1024
      if ( array( i) .ne. (i) ) then
         errors = errors + 1
      end if
   end do

   !$omp target teams distribute parallel do num_teams(3)
   do i = 1, 1024
      array(i) = i
    end do

   do i = 1, 1024
      if ( array( i) .ne. (i) ) then
         errors = errors + 1
      end if
   end do

   print *,"number of errors: ", errors

end program main

! CHECK:  "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}} SPMD-No-Loop mode
! CHECK:  "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}} SPMD mode
! CHECK:  number of errors: 0

