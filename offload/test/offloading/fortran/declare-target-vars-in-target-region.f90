! Offloading test with a target region mapping a declare target Fortran array
! writing some values to it and checking the host correctly receives the
! updates made on the device.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
module test_0
  implicit none
  INTEGER :: arr1(10) = (/0,0,0,0,0,0,0,0,0,0/)
  INTEGER :: arr2(10) = (/0,0,0,0,0,0,0,0,0,0/)
  !$omp declare target link(arr1) enter(arr2)
  INTEGER :: scalar = 1
  !$omp declare target link(scalar)
end module test_0

subroutine test_with_array_link_and_tofrom()
  use test_0
  integer :: i = 1
  integer :: j = 11
  !$omp target map(tofrom:arr1, i, j)
  do while (i <= j)
      arr1(i) = i;
      i = i + 1
  end do
  !$omp end target

  ! CHECK: 1 2 3 4 5 6 7 8 9 10
  PRINT *, arr1(:)
end subroutine test_with_array_link_and_tofrom

subroutine test_with_array_link_only()
  use test_0
  integer :: i = 1
  integer :: j = 11
  !$omp target map(i, j)
      do while (i <= j)
          arr1(i) = i + 1;
          i = i + 1
      end do
  !$omp end target

  ! CHECK: 2 3 4 5 6 7 8 9 10 11
  PRINT *, arr1(:)
end subroutine test_with_array_link_only

subroutine test_with_array_enter_only()
  use test_0
  integer :: i = 1
  integer :: j = 11
  !$omp target map(i, j)
      do while (i <= j)
          arr2(i) = i + 1;
          i = i + 1
      end do
  !$omp end target

  ! CHECK: 0 0 0 0 0 0 0 0 0 0
  PRINT *, arr2(:)
end subroutine test_with_array_enter_only

subroutine test_with_scalar_link_only()
  use test_0
  !$omp target
      scalar = 10
  !$omp end target

  ! CHECK: 10
  PRINT *, scalar
end subroutine test_with_scalar_link_only

program main
  call test_with_array_link_and_tofrom()
  call test_with_array_link_only()
  call test_with_array_enter_only()
  call test_with_scalar_link_only()
end program
