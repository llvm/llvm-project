// RUN: %libomptarget-compilexx-run-and-check-generic
// XFAIL: *

#include <cstdlib>
#include <cstdio>
#include <cassert>

struct S {
  int a;
  int b;
  int c;
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
  v->s0->a = 10;
  v->s0->b = 10;
  v->s0->c = 10;
  v->s1->a = 20;
  v->s1->b = 20;
  v->s1->c = 20;
  v->s2->a = 30;
  v->s2->b = 30;
  v->s2->c = 30;

#pragma omp target map(to: v[:1]) map(tofrom: v->s1->b, v->s1->c, v->s2->b)
  {
    v->s1->b += 3;
    v->s1->c += 5;
    v->s2->b += 7;
  }

  printf ("%d\n", v->s0->a); // CHECK: 10
  printf ("%d\n", v->s0->b); // CHECK: 10
  printf ("%d\n", v->s0->c); // CHECK: 10
  printf ("%d\n", v->s1->a); // CHECK: 20
  printf ("%d\n", v->s1->b); // CHECK: 23
  printf ("%d\n", v->s1->c); // CHECK: 25
  printf ("%d\n", v->s2->a); // CHECK: 30
  printf ("%d\n", v->s2->b); // CHECK: 37
  printf ("%d\n", v->s2->c); // CHECK: 30

  free(v->s0);
  free(v->s1);
  free(v->s2);
  free(v);

  return 0;
}
