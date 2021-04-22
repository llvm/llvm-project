// RUN: %libomptarget-compilexx-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-x86_64-pc-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-nvptx64-nvidia-cuda

// UNSUPPORTED: clang

#include <cstdio>
#include <cstdlib>

typedef struct {
  int a;
  double *b;
} C1;
#pragma omp declare mapper(C1 s) map(to : s.a) map(from : s.b [0:2])

typedef struct {
  int a;
  double *b;
  C1 c;
} C;
#pragma omp declare mapper(C s) map(to : s.a, s.c) map(from : s.b [0:2])

typedef struct {
  int e;
  C f;
  int h;
} D;

int main() {
  constexpr int N = 10;
  D sa[2];
  double x[2], y[2];
  double x1[2], y1[2];
  y[1] = x[1] = 20;

  sa[0].e = 111;
  sa[0].f.a = 222;
  sa[0].f.c.a = 777;
  sa[0].f.b = &x[0];
  sa[0].f.c.b = &x1[0];
  sa[0].h = N;

  sa[1].e = 111;
  sa[1].f.a = 222;
  sa[1].f.c.a = 777;
  sa[1].f.b = &y[0];
  sa[1].f.c.b = &y1[0];
  sa[1].h = N;

  printf("%d %d %d %4.5f %d\n", sa[1].e, sa[1].f.a, sa[1].f.c.a, sa[1].f.b[1],
         sa[1].f.b == &x[0] ? 1 : 0);
  // CHECK: 111 222 777 20.00000 1

  __intptr_t p = reinterpret_cast<__intptr_t>(&y[0]);
#pragma omp target map(tofrom : sa) firstprivate(p)
  {
    printf("%d %d %d\n", sa[1].f.a, sa[1].f.c.a,
           sa[1].f.b == reinterpret_cast<void *>(p) ? 1 : 0);
    // CHECK: 222 777 0
    sa[1].e = 333;
    sa[1].f.a = 444;
    sa[1].f.c.a = 555;
    sa[1].f.b[1] = 40;
  }
  printf("%d %d %d %4.5f %d\n", sa[1].e, sa[1].f.a, sa[1].f.c.a, sa[1].f.b[1],
         sa[1].f.b == &x[0] ? 1 : 0);
  // CHECK: 333 222 777 40.00000 1
}
