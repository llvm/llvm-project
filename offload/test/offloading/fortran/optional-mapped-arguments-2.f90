! OpenMP offloading regression test that checks we do not cause a segfault when
! implicitly mapping a not present optional allocatable function argument and
! utilise it in the target region. No results requiring checking other than
! that the program compiles and runs to completion with no error.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
module mod
  implicit none
contains
  subroutine routine(a, b)
    implicit none
    real(4), allocatable, optional, intent(in) :: a(:)
    real(4), intent(out) :: b(:)
    integer(4) :: i, ia
    if(present(a)) then
       ia = 1
       write(*,*) "a is present"
    else
       ia=0
       write(*,*) "a is not present"
    end if

    !$omp target teams distribute parallel do shared(a,b,ia)
    do i=1,10
       if (ia>0) then
          b(i) = b(i) + a(i)
       end if
    end do

  end subroutine routine

end module mod

program main
  use mod
  implicit none
  real(4), allocatable :: a(:)
  real(4), allocatable :: b(:)
  integer(4) :: i
  allocate(b(10))
  do i=1,10
     b(i)=0
  end do
  !$omp target data map(from: b)

  call routine(b=b)

  !$omp end target data

  deallocate(b)

  print *, "success, no segmentation fault"
end program main

!CHECK: a is not present
!CHECK: success, no segmentation fault
