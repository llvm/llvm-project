! Basic offloading test that makes sure we can use the predominantly host
! pragma threadprivate in the same program as target code
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
program main
    implicit none

    type dtype
       integer :: val(10)
    end type dtype

    integer :: i
    type(dtype), pointer :: pointer1
    type(dtype), pointer :: pointer2=>null()
    integer, dimension(:), pointer :: data_pointer

!$omp threadprivate(pointer2)

nullify(pointer1)
allocate(pointer1)

pointer2=>pointer1
pointer2%val(:)=1
data_pointer=>pointer2%val

!$omp target
   do i = 1, 10
     data_pointer(i) = i
   end do
!$omp end target

print *, data_pointer

end program main

! CHECK: 1 2 3 4 5 6 7 8 9 10
