// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: libc

#include <assert.h>
#include <stdio.h>

#pragma omp declare target to(stdout)

int main() {
  int r = 0;
// CHECK: PASS
#pragma omp target map(from : r)
  { r = fwrite("PASS\n", 1, sizeof("PASS\n") - 1, stdout); }
  assert(r == sizeof("PASS\n") - 1 && "Incorrect number of bytes written");

// CHECK: PASS
#pragma omp target map(from : r) nowait
  { r = fwrite("PASS\n", 1, 5, stdout); }
#pragma omp taskwait
  assert(r == sizeof("PASS\n") - 1 && "Incorrect number of bytes written");
}
