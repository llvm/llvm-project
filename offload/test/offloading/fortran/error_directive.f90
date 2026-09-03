! Test the `error` directive with `at(execution)` inside a target region.
!
! REQUIRES: flang, libc

! flang does not accept -foffload-lto.
! UNSUPPORTED: nvptx64-nvidia-cuda-LTO

! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=51 && \
! RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=NOLOC
! RUN: %libomptarget-compile-fortran-generic -fopenmp-version=51 -g && \
! RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=LOC

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
! Without -g the ident holds no location and is reported as "unknown:0:0", the
! same as the host runtime.

! NOLOC: OMP: unknown:0:0: Encountered user-directed warning: warning message.
! NOLOC: OMP: unknown:0:0: Encountered user-directed warning.

! LOC: {{.*}}error_directive.f90:{{[0-9]+}}:{{[0-9]+}}: Encountered user-directed warning: warning message.
! LOC: {{.*}}error_directive.f90:{{[0-9]+}}:{{[0-9]+}}: Encountered user-directed warning.
