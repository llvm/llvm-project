// RUN: %libomptarget-compilexx-generic -fopenmp-version=51
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// UNSUPPORTED: amdgcn-amd-amdhsa

#include <stdio.h>
#include <stdlib.h>

void foo(int **t1d) {
  int ***t2d = &t1d;
  int ****t3d = &t2d;
  *t1d = (int *)malloc(3 * sizeof(int));
  int j, a = 0, b = 0;

  for (j = 0; j < 3; j++)
    (*t1d)[j] = 0;
#pragma omp target map(tofrom : (*t1d)[0 : 3])
  { (*t1d)[1] = 1; }
  // CHECK: 1
  printf("%d\n", (*t1d)[1]);
#pragma omp target map(tofrom : (**t2d)[0 : 3])
  { (**t2d)[1] = 2; }
  // CHECK: 2
  printf("%d\n", (**t2d)[1]);
#pragma omp target map(tofrom : (***t3d)[0 : 3])
  { (***t3d)[1] = 3; }
  // CHECK: 3
  printf("%d\n", (***t3d)[1]);
#pragma omp target map(tofrom : (**t1d))
  { (*t1d)[0] = 4; }
  // CHECK: 4
  printf("%d\n", (*t1d)[0]);
#pragma omp target map(tofrom : (*(*(t1d + a) + b)))
  { *(*(t1d + a) + b) = 5; }
  // CHECK: 5
  printf("%d\n", *(*(t1d + a) + b));
}

typedef int(T)[3];
void bar() {
  T **a;
  int b[2][3];
  int(*p)[3] = b;
  a = &p;
  for (int i = 0; i < 3; i++) {
    (**a)[1] = i;
  }
#pragma omp target map((**a)[ : 3])
  {
    (**a)[1] = 6;
    // CHECK: 6
    printf("%d\n", (**a)[1]);
  }
}

struct SSA {
  int i;
  SSA *sa;
  SSA() {
    i = 1;
    sa = this;
  }
};

void zoo(int **f, SSA *sa) {
  int *t = *f;
  f = (int **)malloc(sa->i * 4 * sizeof(int));
  t = (int *)malloc(sa->i * sizeof(int));
  *(f + sa->i + 1) = t;
  *(sa->sa->i + *(f + sa->i + 1)) = 4;
  printf("%d\n", *(sa->sa->i + *(1 + sa->i + f)));
#pragma omp target map(sa, *(sa->sa->i + *(1 + sa->i + f)))
  { *(sa->sa->i + *(1 + sa->i + f)) = 7; }
  // CHECK: 7
  printf("%d\n", *(sa->sa->i + *(1 + sa->i + f)));
}

void xoo() {
  int *x = 0;
  SSA *sa = new SSA();
  zoo(&x, sa);
}

void yoo(int **x) {
  *x = (int *)malloc(2 * sizeof(int));
#pragma omp target map(**x)
  {
    **x = 8;
    // CHECK: 8
    printf("%d\n", **x);
  }
#pragma omp target map(*(*x + 1))
  {
    *(*x + 1) = 9;
    // CHECK: 9
    printf("%d\n", *(*x + 1));
  }
}

int main() {
  int *data = 0;
  foo(&data);
  bar();
  xoo();
  yoo(&data);
}
