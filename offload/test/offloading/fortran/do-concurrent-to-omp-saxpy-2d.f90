! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic -fdo-concurrent-to-openmp=device
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic
module saxpymod
   use iso_fortran_env
   public :: saxpy
contains

subroutine saxpy(a, x, y, n, m)
   use iso_fortran_env
   implicit none
   integer,intent(in) :: n, m
   real(kind=real32),intent(in) :: a
   real(kind=real32), dimension(:,:),intent(in) :: x
   real(kind=real32), dimension(:,:),intent(inout) :: y
   integer :: i, j

   do concurrent(i=1:n, j=1:m)
       y(i,j) = a * x(i,j) + y(i,j)
   end do

   write(*,*) "plausibility check:"
   write(*,'("y(1,1) ",f8.6)') y(1,1)
   write(*,'("y(n,m) ",f8.6)') y(n,m)
end subroutine saxpy

end module saxpymod

program main
   use iso_fortran_env
   use saxpymod, ONLY:saxpy
   implicit none

   integer,parameter :: n = 1000, m=10000
   real(kind=real32), allocatable, dimension(:,:) :: x, y
   real(kind=real32) :: a
   integer :: i

   allocate(x(1:n,1:m), y(1:n,1:m))
   a = 2.0_real32
   x(:,:) = 1.0_real32
   y(:,:) = 2.0_real32

   call saxpy(a, x, y, n, m)

   deallocate(x,y)
end program main

! CHECK:  "PluginInterface" device {{[0-9]+}} info: Launching kernel {{.*}}
! CHECK:  plausibility check:
! CHECK:  y(1,1) 4.0
! CHECK:  y(n,m) 4.0
