! An untied taskloop leaves its pattern task untied. The runtime starts that
! pattern task and finishes it immediately without executing the loop body, so
! it briefly becomes the thread's current task and needs td_last_tied
! initialized like any other untied task.
!
! This is written in Fortran on purpose: flang honours the untied clause on
! taskloop, whereas clang currently drops it and emits a tied pattern task, so
! the path is not reachable from C.
!
! It does not fault on its own -- the task scheduling constraint only reads
! td_last_tied when a thread's deque is full -- so this is coverage rather than
! a reproducer: in an assertions-enabled build it exercises the
! KMP_DEBUG_ASSERT() in __kmp_task_start() that guards the initialization.

! RUN: %flang %flags %openmp_flags %s -o %t.exe
! RUN: %t.exe | FileCheck %s

program untied_taskloop
  implicit none
  integer, parameter :: n = 2048, reps = 10
  integer :: a(n), i, r

  a = 0

  do r = 1, reps
    !$omp parallel
    !$omp single
    !$omp taskloop untied grainsize(7)
    do i = 1, n
      a(i) = a(i) + 1
    end do
    !$omp end taskloop

    !$omp taskloop untied num_tasks(13)
    do i = 1, n
      a(i) = a(i) + 1
    end do
    !$omp end taskloop
    !$omp end single
    !$omp end parallel
  end do

  if (any(a /= 2 * reps)) then
    print *, 'failed'
  else
    print *, 'passed'
  end if
end program

! CHECK: passed
