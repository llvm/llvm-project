// RUN: %libomptarget-compile-generic -fopenmp-version=51
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

#include <stdio.h>
int main() {
  short x[10];
  short *xp = &x[0];

  x[1] = 111;

  printf("%d, %p\n", xp[1], &xp[1]);
#pragma omp target data use_device_addr(xp [1:3]) map(tofrom : x)
#pragma omp target is_device_ptr(xp)
  { xp[1] = 222; }
  // CHECK: 222
  printf("%d, %p\n", xp[1], &xp[1]);
}
