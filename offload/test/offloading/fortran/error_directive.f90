! Test the `error` directive with `at(execution)` inside a target region.
!
! REQUIRES: flang, libc

! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=51 && \
! RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

program error_directive
  implicit none

  !$omp target
  !$omp error at(execution) severity(warning) message("warning message")
  !$omp end target

  ! No MESSAGE clause, so the runtime receives a null message pointer.
  !$omp target
  !$omp error at(execution) severity(warning)
  !$omp end target
end program error_directive

! Device output is flushed after host output, so host prints are not checked.

! CHECK: user-directed warning: warning message.
! CHECK: user-directed warning.
