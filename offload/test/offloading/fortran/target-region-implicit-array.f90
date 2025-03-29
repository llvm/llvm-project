! Basic offloading test of a regular array explicitly passed within a target 
! region
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    integer :: x(10) = (/0,0,0,0,0,0,0,0,0,0/)
    integer :: i = 1
    integer :: j = 11

  !$omp target
     do i = 1, j
        x(i) = i;
     end do
  !$omp end target

   PRINT *, x(:)
end program main

! CHECK: 1 2 3 4 5 6 7 8 9 10
