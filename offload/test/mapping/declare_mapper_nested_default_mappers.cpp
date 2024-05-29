// RUN: %libomptarget-compilexx-run-and-check-generic

#include <cstdio>
#include <cstdlib>

typedef struct {
  int a;
  double *b;
} C1;
#pragma omp declare mapper(C1 s) map(to : s.a) map(from : s.b[0 : 2])

typedef struct {
  int a;
  double *b;
  C1 c;
} C;
#pragma omp declare mapper(C s) map(to : s.a, s.c) map(from : s.b[0 : 2])

typedef struct {
  int e;
  C f;
  int h;
} D;

int main() {
  constexpr int N = 10;
  D s;
  s.e = 111;
  s.f.a = 222;
  s.f.c.a = 777;
  double x[2];
  double x1[2];
  x[1] = 20;
  s.f.b = &x[0];
  s.f.c.b = &x1[0];
  s.h = N;

  D *sp = &s;
  D **spp = &sp;

  printf("%d %d %d %4.5f %d\n", spp[0][0].e, spp[0][0].f.a, spp[0][0].f.c.a,
         spp[0][0].f.b[1], spp[0][0].f.b == &x[0] ? 1 : 0);
  // CHECK: 111 222 777 20.00000 1

  int spp00fa = -1, spp00fca = -1, spp00fb_r = -1;
  __intptr_t p = reinterpret_cast<__intptr_t>(&x[0]);
#pragma omp target map(tofrom: spp[0][0]) firstprivate(p)                           \
                   map(from: spp00fa, spp00fca, spp00fb_r)
  {
    spp00fa = spp[0][0].f.a;
    spp00fca = spp[0][0].f.c.a;
    spp00fb_r = spp[0][0].f.b == reinterpret_cast<void *>(p) ? 1 : 0;
    printf("%d %d %d\n", spp00fa, spp00fca, spp00fb_r);
    // XCHECK: 222 777 0
    spp[0][0].e = 333;
    spp[0][0].f.a = 444;
    spp[0][0].f.c.a = 555;
    spp[0][0].f.b[1] = 40;
  }
  printf("%d %d %d\n", spp00fa, spp00fca, spp00fb_r);
  // CHECK: 222 777 0
  printf("%d %d %d %4.5f %d\n", spp[0][0].e, spp[0][0].f.a, spp[0][0].f.c.a,
         spp[0][0].f.b[1], spp[0][0].f.b == &x[0] ? 1 : 0);
  // CHECK: 333 222 777 40.00000 1
}
