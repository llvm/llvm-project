// RUN: %libomptarget-compile-run-and-check-generic
// RUN: %libomptarget-compile-generic -DOUT_OF_BOUNDS
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic --check-prefix=CHECK-OOB

#include <stdio.h>

int x[10];
typedef struct {
  int y;
  int *p;
} S;

#ifdef OUT_OF_BOUNDS
// s.p[0:20] extends beyond the mapped region x[0:10]; present check should
// fail.
// NOTE: While OpenMP 6.0:296 implies that the present "motion" modifier should
// not be propagated to s.p[0:20], it is under discussion whether that was
// intentional. Not propagating it would require treating the present modifier
// differently for data-motion clauses (to/from) vs. map clauses.
#pragma omp declare mapper(S s) map(s.y, s.p[0 : 20])
#else
#pragma omp declare mapper(S s) map(s.y, s.p[0 : 2])
#endif
S s;

void f1() {
#pragma omp target update to(present : s)

#pragma omp target data use_device_addr(s, x)
#pragma omp target has_device_addr(s, x)
  {
    s.y = s.y + 222;
    x[0] = x[0] + 222;
  }
}

int main() {
  x[0] = 111;
  s.y = 111;
  s.p = &x[0];

  // CHECK-OOB: addr=0x[[#%x,HOST_ADDR:]], size=[[#%u,SIZE:]]
  fprintf(stderr, "addr=%p, size=%zu\n", &s.p[0], 20 * sizeof(s.p[0]));

#pragma omp target data map(from : s.y, x)
  {
    f1();
  }

  printf("%d %d\n", x[0], s.y); // CHECK: 333 333
}
