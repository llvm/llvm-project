// RUN: %libomptarget-compilexx-generic -O3 && %libomptarget-run-generic

#include <stdio.h>

// CHECK: rx: 16, ry: 16;
// CHECK: rx: 16, ry: 16;
// CHECK: rx: 16, ry: 16;
// CHECK: rx: 16, ry: 16;

template <bool Aligned> void test() {
  printf("Test %saligned firstprivate\n", Aligned ? "" : "non-");
  char z1[3 + Aligned], z2[3 + Aligned];
  int x[4];
  int y[4];
  y[0] = y[1] = y[2] = y[3] = 4;
  x[0] = x[1] = x[2] = x[3] = 4;
  int rx = -1, ry = -1;
#pragma omp target firstprivate(z1, y, z2) map(from : ry, rx) map(to : x)
  {
    ry = (y[0] + y[1] + y[2] + y[3]);
    rx = (x[0] + x[1] + x[2] + x[3]);
  }
  printf(" rx:%i, ry:%i\n", rx, ry);
#pragma omp target firstprivate(z1, y, z2) map(from : ry, rx) map(to : x)
  {
    z1[2] += 5;
    ry = (y[0] + y[1] + y[2] + y[3]);
    rx = (x[0] + x[1] + x[2] + x[3]);
    z2[2] += 7;
  }
  printf(" rx:%i, ry:%i\n", rx, ry);
}

int main() {
  test<true>();
  test<false>();
}
