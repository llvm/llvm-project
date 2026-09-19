// Test the `error` directive with `at(execution)` inside a target region.
//
// REQUIRES: libc

// RUN: %libomptarget-compile-generic -fopenmp-version=51 && \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: %libomptarget-compile-generic -fopenmp-version=51 -DFATAL && \
// RUN:   %libomptarget-run-fail-generic

#include <stdio.h>

int main(void) {
#ifdef FATAL
#pragma omp target
  {
#pragma omp error at(execution) severity(fatal) message("fatal message")
  }
  printf("unreachable\n");
#else
#pragma omp target
  {
#pragma omp error at(execution) severity(warning) message("warning message")
  }

  // No MESSAGE clause, so the runtime receives a null message pointer.
#pragma omp target
  {
#pragma omp error at(execution) severity(warning)
  }
#endif
  return 0;
}

// Device output is flushed after host output, so host prints are not checked.
// The fatal case checks only the exit status: its message is lost when the trap
// aborts before the buffered stdout is flushed.
//
// clang fills in the ident location from the AST, so it is present without -g

// CHECK: {{.*}}error_directive.c:{{[0-9]+}}:{{[0-9]+}}: Encountered user-directed warning: warning message.
// CHECK: {{.*}}error_directive.c:{{[0-9]+}}:{{[0-9]+}}: Encountered user-directed warning.
