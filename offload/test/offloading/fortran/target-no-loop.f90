! REQUIRES: flang
! REQUIRES: gpu

! RUN: %libomptarget-compile-fortran-generic -O3  -fopenmp-assume-threads-oversubscription -fopenmp-assume-teams-oversubscription
! RUN: env LIBOMPTARGET_INFO=16 OMP_NUM_TEAMS=16 OMP_TEAMS_THREAD_LIMIT=16 %libomptarget-run-generic 2>&1 | %fcheck-generic
! XFAIL: intelgpu
function check_errors(array) result (errors)
   integer, intent(in) :: array(1024)
   integer :: errors
   integer :: i
   errors = 0
   do i = 1, 1024
      if ( array( i) .ne. (i) ) then
         errors = errors + 1
      end if
   end do
end function

program main
   use omp_lib
   implicit none
   integer :: i,j,red
   integer :: array(1024), errors = 0
   array = 1

   ! No-loop kernel
   !$omp target teams distribute parallel do
   do i = 1, 1024
      array(i) = i
   end do
   errors = errors + check_errors(array)

   ! SPMD kernel (num_teams clause blocks promotion to no-loop)
   array = 1
   !$omp target teams distribute parallel do num_teams(3)
   do i = 1, 1024
      array(i) = i
   end do

   errors = errors + check_errors(array)

   ! No-loop kernel
   array = 1
   !$omp target teams distribute parallel do num_threads(64)
   do i = 1, 1024
      array(i) = i
    end do

   errors = errors + check_errors(array)

   ! SPMD kernel
   array = 1
   !$omp target parallel do
   do i = 1, 1024
      array(i) = i
   end do

   errors = errors + check_errors(array)

   ! Generic kernel
   array = 1
   !$omp target teams distribute
   do i = 1, 1024
      array(i) = i
   end do

   errors = errors + check_errors(array)

   ! SPMD kernel (reduction clause blocks promotion to no-loop)
   array = 1
   red =0
   !$omp target teams distribute parallel do reduction(+:red)
   do i = 1, 1024
      red = red + array(i)
   end do

   if (red .ne. 1024) then
      errors = errors + 1
   end if

   print *,"number of errors: ", errors

end program main

! CHECK:  "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}} SPMD-No-Loop mode
! CHECK:  info: #Args: 3 Teams x Thrds:   64x  16
! CHECK:  "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}} SPMD mode
! CHECK:  info: #Args: 3 Teams x Thrds:   3x  16 {{.*}}
! CHECK:  "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}} SPMD-No-Loop mode
! CHECK:  info: #Args: 3 Teams x Thrds:   64x  16 {{.*}}
! CHECK:  "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}} SPMD mode
! CHECK:  info: #Args: 3 Teams x Thrds:   1x  16
! CHECK:  "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}} Generic mode
! CHECK:  info: #Args: 3 Teams x Thrds:   16x  16 {{.*}}
! CHECK:  "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}} SPMD mode
! CHECK:  info: #Args: 4 Teams x Thrds:   16x  16 {{.*}}
! CHECK:  number of errors: 0

