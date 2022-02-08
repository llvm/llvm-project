// RUN: %libomptarget-compilexx-run-and-check-generic

// Wrong results on amdgpu
// XFAIL: amdgcn-amd-amdhsa
// XFAIL: amdgcn-amd-amdhsa-newDriver

#include <stdio.h>
#include <stdint.h>

// CHECK: before: [[V1:111]] [[V2:222]] [[PX:0x[^ ]+]] [[PY:0x[^ ]+]]
// CHECK: lambda: [[V1]] [[V2]] [[PX_TGT:0x[^ ]+]] 0x{{.*}}
// CHECK: tgt   : [[V2]] [[PX_TGT]] 1
// CHECK: out   : [[V2]] [[V2]] [[PX]] [[PY]]

int main() {
  int x[10];
  long y[8];
  x[1] = 111;
  y[1] = 222;

  auto lambda = [&x, y]() {
    printf("lambda: %d %ld %p %p\n", x[1], y[1], &x[0], &y[0]);
    x[1] = y[1];
  };

  printf("before: %d %ld %p %p\n", x[1], y[1], &x[0], &y[0]);

  intptr_t xp = (intptr_t) &x[0];
#pragma omp target firstprivate(xp)
  {
    lambda();
    printf("tgt   : %d %p %d\n", x[1], &x[0], (&x[0] != (int*) xp));
  }
  printf("out   : %d %ld %p %p\n", x[1], y[1], &x[0], &y[0]);

  return 0;
}

