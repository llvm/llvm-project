! RUN: %libomptarget-compile-fortran-run-and-check-generic
! REQUIRES: flang, amdgpu


program main
  implicit none
  integer, parameter :: nnn = 1000
  integer :: aaa(nnn), bbb
  integer :: x, y, got
  integer :: i

  x = 42
  got = -1
  !$omp target firstprivate(x) map(from:got)
    got = x + 1
  !$omp end target

  if (got .ne. 43) then
    print *, "FAIL target firstprivate"
    stop 1
  end if
  ! CHECK: PASS target firstprivate
  print *, "PASS target firstprivate"

  y = 10
  got = -1
  !$omp target firstprivate(y) nowait map(from:got)
    got = y + 1
  !$omp end target
  !$omp taskwait

  if (got .ne. 11) then
    print *, "FAIL target firstprivate nowait"
    stop 1
  end if
  ! CHECK: PASS target firstprivate nowait
  print *, "PASS target firstprivate nowait"

  aaa = 0
  bbb = 1
  !$omp target teams distribute parallel do firstprivate(bbb)
  do i = 1, nnn
    aaa(i) = bbb
  end do

  if (sum(abs(aaa)) .ne. nnn * bbb .or. any(aaa .ne. 1)) then
    print *, "FAIL target teams distribute parallel do firstprivate"
    stop 1
  end if
  ! CHECK: PASS target teams distribute parallel do firstprivate
  print *, "PASS target teams distribute parallel do firstprivate"
end program main
