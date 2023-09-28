// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: libc

#include <stdio.h>
#include <stdlib.h>

#pragma omp declare target to(malloc)
#pragma omp declare target to(free)

int main() {
  unsigned h_x;
  unsigned *d_x;
#pragma omp target map(from : d_x)
  {
    d_x = malloc(sizeof(unsigned));
    *d_x = 1;
  }

#pragma omp target is_device_ptr(d_x) map(from : h_x)
  { h_x = *d_x; }

#pragma omp target is_device_ptr(d_x)
  { free(d_x); }

  // CHECK: PASS
  if (h_x == 1)
    fputs("PASS\n", stdout);
}
