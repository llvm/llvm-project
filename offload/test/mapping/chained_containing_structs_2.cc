// RUN: %libomptarget-compilexx-run-and-check-generic
// XFAIL: *

#include <cstdlib>
#include <cstdio>
#include <cassert>

struct R {
  int d;
  int e;
  int f;
};

struct S {
  R *r0;
  R *r1;
  R *r2;
};

struct T {
  S *s0;
  S *s1;
  S *s2;
};

int main() {
  T *v = (T *) malloc (sizeof(T));

  v->s0 = (S *) malloc (sizeof(S));
  v->s1 = (S *) malloc (sizeof(S));
  v->s2 = (S *) malloc (sizeof(S));

  v->s0->r0 = (R *) calloc (1, sizeof(R));
  v->s0->r1 = (R *) calloc (1, sizeof(R));
  v->s0->r2 = (R *) calloc (1, sizeof(R));

  v->s1->r0 = (R *) calloc (1, sizeof(R));
  v->s1->r1 = (R *) calloc (1, sizeof(R));
  v->s1->r2 = (R *) calloc (1, sizeof(R));

  v->s2->r0 = (R *) calloc (1, sizeof(R));
  v->s2->r1 = (R *) calloc (1, sizeof(R));
  v->s2->r2 = (R *) calloc (1, sizeof(R));

  #pragma omp target map(to: v->s1, v->s2, *v->s1, v->s1->r1, *v->s2, v->s2->r0) \
                     map(tofrom: v->s1->r1->d, v->s1->r1->e, v->s1->r2->d, v->s1->r2->f, v->s2->r0->e)
  {
    v->s1->r1->d += 3;
    v->s1->r1->e += 5;
    v->s1->r2->d += 7;
    v->s1->r2->f += 9;
    v->s2->r0->e += 11;
  }

  printf ("%d\n", v->s1->r1->d); // CHECK: 3
  printf ("%d\n", v->s1->r1->e); // CHECK: 5
  printf ("%d\n", v->s1->r2->d); // CHECK: 7
  printf ("%d\n", v->s1->r2->f); // CHECK: 9
  printf ("%d\n", v->s2->r0->e); // CHECK: 11

  free(v->s0->r0);
  free(v->s0->r1);
  free(v->s0->r2);
  free(v->s1->r0);
  free(v->s1->r1);
  free(v->s1->r2);
  free(v->s2->r0);
  free(v->s2->r1);
  free(v->s2->r2);
  free(v->s0);
  free(v->s1);
  free(v->s2);
  free(v);

  return 0;
}
