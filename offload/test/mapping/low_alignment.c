// RUN: %libomptarget-compilexx-run-and-check-generic

#include <omp.h>
#include <stdio.h>

int main() {
  struct S {
    int i;
    int j;
  } s;
  s.i = 20;
  s.j = 30;
#pragma omp target data map(tofrom : s)
  {
#pragma omp target map(from : s.i, s.j)
    {
      s.i = 21;
      s.j = 31;
    }
  }
  if (s.i == 21 && s.j == 31)
    printf("PASS 1\n");
  // CHECK: PASS 1

  struct T {
    int a;
    int b;
    int c;
    int d;
    int i;
    int j;
  } t;
  t.a = 10;
  t.i = 20;
  t.j = 30;
#pragma omp target data map(from : t.i, t.j)
  {
#pragma omp target map(from : t.a)
    {
      t.a = 11;
      t.i = 21;
      t.j = 31;
    }
  }
  if (t.a == 11 && t.i == 21 && t.j == 31)
    printf("PASS 2\n");
  // CHECK: PASS 2
  return 0;
}
