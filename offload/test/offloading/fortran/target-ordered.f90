! RUN: %libomptarget-compile-run-and-check-generic
! REQUIRES: flang
! REQUIRES: gpu

! An ordered region must execute in iteration order. Record the order in which
! iterations enter it; any inversion means the guarantee was not honoured.
program ordered_target
  implicit none
  integer, parameter :: n = 64
  integer :: seq(n), pos, i, bad
  pos = 0
  seq = -1
  !$omp target parallel do ordered map(tofrom:seq,pos)
  do i = 1, n
     !$omp ordered
     pos = pos + 1
     seq(pos) = i
     !$omp end ordered
  end do
  bad = 0
  do i = 1, n
     if (seq(i) /= i) bad = bad + 1
  end do
  ! CHECK: out of order: 0
  print '(A,I4)', "out of order: ", bad
end program
