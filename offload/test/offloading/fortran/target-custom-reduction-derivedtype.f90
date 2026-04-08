! Basic offloading test with custom OpenMP reduction on derived type
! REQUIRES: flang, amdgpu
!
! RUN: %libomptarget-compile-fortran-generic
! RUN: env LIBOMPTARGET_INFO=16 %libomptarget-run-generic 2>&1 | %fcheck-generic
module maxtype_mod
  implicit none

  type maxtype
     integer::sumval
     integer::maxval
  end type maxtype

contains

  subroutine initme(x,n)
    type(maxtype) :: x,n
    x%sumval=0
    x%maxval=0
  end subroutine initme

  function mycombine(lhs, rhs)
    type(maxtype) :: lhs, rhs
    type(maxtype) :: mycombine
    mycombine%sumval = lhs%sumval + rhs%sumval
    mycombine%maxval = max(lhs%maxval, rhs%maxval)
  end function mycombine

end module maxtype_mod

program main
  use maxtype_mod
  implicit none

  integer :: n = 100
  integer :: i
  integer :: error = 0
  type(maxtype) :: x(100)
  type(maxtype) :: res
  integer :: expected_sum, expected_max

!$omp declare reduction(red_add_max:maxtype:omp_out=mycombine(omp_out,omp_in)) initializer(initme(omp_priv,omp_orig))

  ! Initialize array with test data
  do i = 1, n
    x(i)%sumval = i
    x(i)%maxval = i
  end do

  ! Initialize reduction variable
  res%sumval = 0
  res%maxval = 0

  ! Perform reduction in target region
  !$omp target parallel do map(to:x) reduction(red_add_max:res)
  do i = 1, n
    res = mycombine(res, x(i))
  end do
  !$omp end target parallel do

  ! Compute expected values
  expected_sum = 0
  expected_max = 0
  do i = 1, n
    expected_sum = expected_sum + i
    expected_max = max(expected_max, i)
  end do

  ! Check results
  if (res%sumval /= expected_sum) then
    error = 1
  endif

  if (res%maxval /= expected_max) then
    error = 1
  endif

  if (error == 0) then
    print *,"PASSED"
  else
    print *,"FAILED"
  endif

end program main

! CHECK:  PluginInterface device {{[0-9]+}} info: Launching kernel {{.*}}
! CHECK:  PASSED

