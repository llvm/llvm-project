// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: libc

#include <stdio.h>

#pragma omp declare target to(stdout)

int main() {
// CHECK: PASS
#pragma omp target
  { fputs("PASS\n", stdout); }

// CHECK: PASS
#pragma omp target nowait
  { fputs("PASS\n", stdout); }

// CHECK: PASS
#pragma omp target nowait
  { fputs("PASS\n", stdout); }

#pragma omp taskwait

// CHECK: PASS
// CHECK: PASS
// CHECK: PASS
// CHECK: PASS
// CHECK: PASS
// CHECK: PASS
// CHECK: PASS
// CHECK: PASS
#pragma omp target teams num_teams(4)
#pragma omp parallel num_threads(2)
  { puts("PASS\n"); }
}
