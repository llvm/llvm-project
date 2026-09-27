! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic -fdo-concurrent-to-openmp=device
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic

module min_reduce_mod
   implicit none
   public :: min_reduce
contains

subroutine min_reduce(arr, min_val, n)
   implicit none
   integer, intent(in) :: n
   real, dimension(:), intent(in) :: arr
   real :: min_val
   integer :: i

   do concurrent(i=1:n) reduce(min:min_val)
       min_val = min(min_val, arr(i))
   end do

   print *, 'min_val after reduction =', min_val
end subroutine min_reduce

end module min_reduce_mod

program main
   use min_reduce_mod, only: min_reduce
   implicit none

   integer, parameter :: n = 10
   real :: arr(n)
   real :: min_val

   arr = (/ 200.0, 150.0, 80.0, 50.0, 300.0, 25.0, 175.0, 60.0, 400.0, 90.0 /)
   min_val = 100.0

   call min_reduce(arr, min_val, n)

   if (min_val == 25.0) then
       print *, 'PASS'
   else
       print *, 'FAIL: expected 25.0, got', min_val
   end if
end program main

! CHECK:  PluginInterface device {{[0-9]+}} info: Launching kernel {{.*}}
! CHECK:  PASS
