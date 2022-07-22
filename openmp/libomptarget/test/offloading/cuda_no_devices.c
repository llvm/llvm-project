// The CUDA plugin used to complain on stderr when no CUDA devices were enabled,
// and then it let the application run anyway.  Check that there's no such
// complaint anymore, especially when the user isn't targeting CUDA.

// RUN: %libomptarget-compile-generic
// RUN: env CUDA_VISIBLE_DEVICES= \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic

#include <stdio.h>

// CHECK-NOT: {{.}}
//     CHECK: Hello World: 4
// CHECK-NOT: {{.}}
int main() {
  int x = 0;
  #pragma omp target teams num_teams(2) reduction(+:x)
  x += 2;
  printf("Hello World: %d\n", x);
  return 0;
}
