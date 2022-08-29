// RUN: %libomptarget-compilexx-run-and-check-generic

// Wrong results on amdgpu
// XFAIL: amdgcn-amd-amdhsa
// XFAIL: amdgcn-amd-amdhsa-LTO

#include <cstdio>
#include <cstdlib>

typedef struct {
  int a;
  double *b;
} C;
#pragma omp declare mapper(id1 : C s) map(to : s.a) map(from : s.b [0:2])

typedef struct {
  int e;
  C f;
  int h;
  short *g;
} D;
#pragma omp declare mapper(default                                             \
                           : D r) map(from                                     \
                                      : r.e) map(mapper(id1), tofrom           \
                                                 : r.f) map(tofrom             \
                                                            : r.g [0:r.h])

int main() {
  constexpr int N = 10;
  D s;
  s.e = 111;
  s.f.a = 222;
  double x[2];
  x[1] = 20;
  short y[N];
  y[1] = 30;
  s.f.b = &x[0];
  s.g = &y[0];
  s.h = N;

  D *sp = &s;
  D **spp = &sp;

  printf("%d %d %4.5f %d %d %d\n", spp[0][0].e, spp[0][0].f.a, spp[0][0].f.b[1],
         spp[0][0].f.b == &x[0] ? 1 : 0, spp[0][0].g[1],
         spp[0][0].g == &y[0] ? 1 : 0);
  // CHECK: 111 222 20.00000 1 30 1

  __intptr_t p = reinterpret_cast<__intptr_t>(&x[0]),
             p1 = reinterpret_cast<__intptr_t>(&y[0]);
#pragma omp target map(tofrom : spp[0][0]) firstprivate(p, p1)
  {
    printf("%d %d %d %d\n", spp[0][0].f.a,
           spp[0][0].f.b == reinterpret_cast<void *>(p) ? 1 : 0, spp[0][0].g[1],
           spp[0][0].g == reinterpret_cast<void *>(p1) ? 1 : 0);
    // CHECK: 222 0 30 0
    spp[0][0].e = 333;
    spp[0][0].f.a = 444;
    spp[0][0].f.b[1] = 40;
    spp[0][0].g[1] = 50;
  }
  printf("%d %d %4.5f %d %d %d\n", spp[0][0].e, spp[0][0].f.a, spp[0][0].f.b[1],
         spp[0][0].f.b == &x[0] ? 1 : 0, spp[0][0].g[1],
         spp[0][0].g == &y[0] ? 1 : 0);
  // CHECK: 333 222 40.00000 1 50 1
}
