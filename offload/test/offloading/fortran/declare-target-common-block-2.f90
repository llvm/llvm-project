! Make sure we handle both forms of common block with variations of 
! declare target correctly.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-run-and-check-generic
block data init
  implicit none
  
  ! Declare variables that will be in the COMMON block
  real :: x(5)
  integer :: count
  real :: pi
  
  ! Associate them with a named COMMON block
  common /myblock/ x, count, pi
  
  ! Initialize using DATA statements
  data x(1), x(2), x(3), x(4), x(5)  / 1.0, 2.0, 3.0, 4.0, 5.0 /
  data count / 42 /
  data pi / 3.14159 /
  
end block data

program prog
    implicit none
    real :: x(5), pi
    integer :: count, i
    common /myblock/ x, count, pi
    common/cblock_a/var_a
    real(kind=8), dimension(5) :: var_a
   !$omp declare target(/cblock_a/)
   !$omp declare target link(/myblock/)

 !$omp target map(tofrom: /myblock/) map(always, tofrom: /cblock_a/)
    do i = 1, 5
      var_a(i) = i
    end do

    count = 2000
    pi = 1.233
    do i = 1, 5
      x(i) = x(i) + 1
    end do
  !$omp end target

  print *, var_a
  print *, count
  print *, pi
  print *, x
end program

! CHECK: 1. 2. 3. 4. 5.
! CHECK: 2000
! CHECK: 1.233
! CHECK: 2. 3. 4. 5. 6.
