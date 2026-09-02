! Test `severity(fatal)` on the `error` directive with `at(execution)` inside a
! target region: execution aborts.
!
! Only the exit status is checked: the message is lost when the trap aborts
! before the buffered stdout is flushed. error_directive.f90 covers the text.
!
! REQUIRES: flang, libc

! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=51 && \
! RUN:   %libomptarget-run-fail-generic

program error_directive_fatal
  implicit none

  !$omp target
  !$omp error at(execution) severity(fatal) message("fatal message")
  !$omp end target

  print *, "unreachable"
end program error_directive_fatal
