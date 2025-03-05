// RUN: %libomptarget-compile-generic -fopenmp-version=51
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// UNSUPPORTED: amdgcn-amd-amdhsa

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define FROM 64
#define LENGTH 128

void foo() {
  const int device_id = omp_get_default_device();
  float *A;
  A = (float *)omp_target_alloc((FROM + LENGTH) * sizeof(float), device_id);

  float *A_dev = NULL;
#pragma omp target has_device_addr(A[FROM : LENGTH]) map(A_dev)
  { A_dev = A; }
  // CHECK: Success
  if (A_dev == NULL || A_dev != A)
    fprintf(stderr, "Failure %p %p \n", A_dev, A);
  else
    fprintf(stderr, "Success\n");
}

void bar() {
  short x[10];
  short *xp = &x[0];

  x[1] = 111;
#pragma omp target data map(tofrom : xp[0 : 2]) use_device_addr(xp[0 : 2])
#pragma omp target has_device_addr(xp[0 : 2])
  {
    xp[1] = 222;
    // CHECK: 222
    printf("%d %p\n", xp[1], &xp[1]);
  }
  // CHECK: 222
  printf("%d %p\n", xp[1], &xp[1]);
}

void moo() {
  short *b = malloc(sizeof(short));
  b = b - 1;

  b[1] = 111;
#pragma omp target data map(tofrom : b[1]) use_device_addr(b[1])
#pragma omp target has_device_addr(b[1])
  {
    b[1] = 222;
    // CHECK: 222
    printf("%hd %p %p %p\n", b[1], b, &b[1], &b);
  }
  // CHECK: 222
  printf("%hd %p %p %p\n", b[1], b, &b[1], &b);
}

void zoo() {
  short x[10];
  short *(xp[10]);
  xp[1] = &x[0];
  short **xpp = &xp[0];

  x[1] = 111;
#pragma omp target data map(tofrom : xpp[1][1]) use_device_addr(xpp[1][1])
#pragma omp target has_device_addr(xpp[1][1])
  {
    xpp[1][1] = 222;
    // CHECK: 222
    printf("%d %p %p\n", xpp[1][1], xpp[1], &xpp[1][1]);
  }
  // CHECK: 222
  printf("%d %p %p\n", xpp[1][1], xpp[1], &xpp[1][1]);
}
void xoo() {
  short a[10], b[10];
  a[1] = 111;
  b[1] = 111;
#pragma omp target data map(to : a[0 : 2], b[0 : 2]) use_device_addr(a, b)
#pragma omp target has_device_addr(a) has_device_addr(b[0])
  {
    a[1] = 222;
    b[1] = 222;
    // CHECK: 222 222
    printf("%hd %hd %p %p %p\n", a[1], b[1], &a, b, &b);
  }
  // CHECK:111
  printf("%hd %hd %p %p %p\n", a[1], b[1], &a, b, &b); // 111 111 p1d p2d p3d
}
int main() {
  foo();
  bar();
  moo();
  zoo();
  xoo();
  return 0;
}
