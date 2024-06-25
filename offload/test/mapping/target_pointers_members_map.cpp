// RUN: %libomptarget-compilexx-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-x86_64-pc-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-nvptx64-nvidia-cuda

#include <cstdio>
#include <cstdlib>

typedef struct {
  short *a;
  long d1, d2;
} DV_A;

typedef struct {
  DV_A b;
  long d3;
} C;

typedef struct {
  C *c;
  long d4, d5;
} DV_B;

int main() {

  short arr1[10] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  short arr2[10] = {20, 31, 22, 23, 24, 25, 26, 27, 28, 29};

  C c1[2];
  c1[0].b.a = (short *)arr1;
  c1[1].b.a = (short *)arr2;
  c1[0].b.d1 = 111;

  DV_B dvb1;
  dvb1.c = (C *)&c1;

  // CHECK: 10 111
  printf("%d %ld %p %p %p %p\n", dvb1.c[0].b.a[0], dvb1.c[0].b.d1, &dvb1,
         &dvb1.c[0], &dvb1.c[0].b, &dvb1.c[0].b.a[0]);
#pragma omp target map(to : dvb1, dvb1.c[0 : 2])                               \
    map(tofrom : dvb1.c[0].b.a[0 : 10], dvb1.c[1].b.a[0 : 10])
  {
    // CHECK: 10 111
    printf("%d %ld %p %p %p %p\n", dvb1.c[0].b.a[0], dvb1.c[0].b.d1, &dvb1,
           &dvb1.c[0], &dvb1.c[0].b, &dvb1.c[0].b.a[0]);
    dvb1.c[0].b.a[0] = 333;
    dvb1.c[0].b.d1 = 444;
  }
  // CHECK: 333 111
  printf("%d %ld %p %p %p %p\n", dvb1.c[0].b.a[0], dvb1.c[0].b.d1, &dvb1,
         &dvb1.c[0], &dvb1.c[0].b, &dvb1.c[0].b.a[0]);
}
