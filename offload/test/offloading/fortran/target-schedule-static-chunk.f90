! schedule(static,C) prescribes which thread runs which iteration: thread T
! owns [T*C, T*C+C). Record the thread that ran each iteration and compare.

! RUN: %libomptarget-compile-run-and-check-generic
! REQUIRES: flang
! REQUIRES: gpu

program target_schedule_static_chunk
  use omp_lib
  implicit none
  integer, parameter :: n = 32, nt = 8, chunk = 4
  integer :: tid(n), i, bad

  tid = -1
  !$omp target teams distribute parallel do num_teams(1) thread_limit(nt) &
  !$omp&        num_threads(nt) schedule(static,chunk) map(tofrom:tid)
  do i = 1, n
     tid(i) = omp_get_thread_num()
  end do

  bad = 0
  do i = 1, n
     if (tid(i) /= mod((i-1)/chunk, nt)) bad = bad + 1
  end do

  ! CHECK: misplaced: 0
  print '(A,I0)', "misplaced: ", bad
end program
