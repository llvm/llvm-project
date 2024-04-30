// RUN: %libomptarget-compilexx-run-and-check-generic

#include <stdint.h>
#include <stdio.h>

// CHECK: before: [[V1:111]] [[V2:222]] [[PX:0x[^ ]+]] [[PY:0x[^ ]+]]
// CHECK: lambda: [[V1]] [[V2]] [[PX_TGT:0x[^ ]+]] 0x{{.*}}
// CHECK: tgt   : [[V2]] [[PX_TGT]] 1
// CHECK: out   : [[V2]] [[V2]] [[PX]] [[PY]]

#pragma omp begin declare target
int a = -1, *c;
long b = -1;
const long *d;
int e = -1, *f, g = -1;
#pragma omp end declare target

int main() {
  int x[10];
  long y[8];
  x[1] = 111;
  y[1] = 222;

  auto lambda = [&x, y]() {
    a = x[1];
    b = y[1];
    c = &x[0];
    d = &y[0];
    printf("lambda: %d %ld %p %p\n", x[1], y[1], &x[0], &y[0]);
    x[1] = y[1];
  };
  printf("before: %d %ld %p %p\n", x[1], y[1], &x[0], &y[0]);

  intptr_t xp = (intptr_t)&x[0];
#pragma omp target firstprivate(xp)
  {
    lambda();
    e = x[1];
    f = &x[0];
    g = (&x[0] != (int *)xp);
    printf("tgt   : %d %p %d\n", x[1], &x[0], (&x[0] != (int *)xp));
  }
#pragma omp target update from(a, b, c, d, e, f, g)
  printf("lambda: %d %ld %p %p\n", a, b, c, d);
  printf("tgt   : %d %p %d\n", e, f, g);
  printf("out   : %d %ld %p %p\n", x[1], y[1], &x[0], &y[0]);

  return 0;
}
