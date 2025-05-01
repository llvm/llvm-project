!REQUIRES: flang, amdgpu

!RUN: %libomptarget-compile-fortran-run-and-check-generic

subroutine f
  type :: t1
    integer :: x, y, z
  end type

  integer, parameter :: n = 9
  type(t1) :: b(n)

  integer :: i
  do i = 1, n
    b(i)%x = 0
    b(i)%y = 0
    b(i)%z = 0
  enddo

  !$omp target data map(tofrom: b(1:3)) use_device_addr(b)
  !$omp target has_device_addr(b(2)%x)
    b(2)%x = 1
  !$omp end target
  !$omp end target data
  print *, "b1", b
end

subroutine g
  type :: t1
    integer :: x(3), y(7), z(5)
  end type

  integer, parameter :: n = 5
  type(t1) :: b(n)

  integer :: i
  do i = 1, n
    b(i)%x = 0
    b(i)%y = 0
    b(i)%z = 0
  enddo

  !$omp target data map(tofrom: b(1:3)) use_device_addr(b)
  !$omp target has_device_addr(b(2)%x)
    b(2)%x(3) = 1
  !$omp end target
  !$omp end target data
  print *, "b2", b
end

call f()
call g()
end

!CHECK: b1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
!CHECK: b2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

