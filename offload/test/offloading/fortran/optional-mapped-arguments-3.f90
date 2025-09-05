! OpenMP offloading test that checks we do not cause a segfault when mapping
! optional function arguments (present or otherwise). No results requiring
! checking other than that the program compiles and runs to completion with no
! error. This particular variation checks that we're correctly emitting the
! load/store in both branches mapping the input array.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
module reproducer_mod
contains
   subroutine branching_target_call(dt, switch)
      implicit none
      real(4), dimension(:), intent(inout) :: dt
      logical, intent(in) :: switch
      integer :: dim, idx

      dim = size(dt)
      if (switch) then
!$omp target teams distribute parallel do
         do idx = 1, dim
            dt(idx) = 20
         end do
      else
!$omp target teams distribute parallel do
         do idx = 1, dim
            dt(idx) = 30
         end do
      end if
   end subroutine branching_target_call
end module reproducer_mod

program reproducer
   use reproducer_mod
   implicit none
   real(4), dimension(:), allocatable :: dt
   integer :: n = 21312
   integer :: i

   allocate (dt(n))
   call branching_target_call(dt, .FALSE.)
   call branching_target_call(dt, .TRUE.)
   print *, "PASSED"
end program reproducer

! CHECK: PASSED
