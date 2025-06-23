// RUN: %libomptarget-compilexx-run-and-check-generic

#include <omp.h>
#include <stdio.h>

int x[10];
int *p;

void f1() {
  p = &x[0];
  p[0] = 111;
  p[1] = 222;
  p[2] = 333;
  p[3] = 444;

#pragma omp target enter data map(to : p)
#pragma omp target enter data map(to:p[0 : 5])

  int **p_mappedptr = (int **)omp_get_mapped_ptr(&p, omp_get_default_device());
  int *p0_mappedptr =
      (int *)omp_get_mapped_ptr(&p[0], omp_get_default_device());

  printf("p_mappedptr %s null\n", p_mappedptr == (int **)NULL ? "==" : "!=");
  printf("p0_mappedptr %s null\n", p0_mappedptr == (int *)NULL ? "==" : "!=");

// CHECK: p_mappedptr != null
// CHECK: p0_mappedptr != null

// p is predetermined firstprivate, so its address will be different from
// the mapped address for this construct. So, any changes to p within the
// region will not be visible after the construct.
#pragma omp target map(p[0]) firstprivate(p_mappedptr, p0_mappedptr)
  {
    printf("%d %d %d\n", p[0], p_mappedptr == &p, p0_mappedptr == &p[0]);
    // CHECK: 111 0 1
    p++;
  }

// For the remaining constructs, p is not firstprivate, so its address will
// be the same as the mapped address, and changes to p will be visible to any
// subsequent regions.
#pragma omp target map(to : p[0], p) firstprivate(p_mappedptr, p0_mappedptr)
  {
    printf("%d %d %d\n", p[0], p_mappedptr == &p, p0_mappedptr == &p[0]);
    // CHECK: 111 1 1
    p++;
  }

#pragma omp target map(to : p, p[0]) firstprivate(p_mappedptr, p0_mappedptr)
  {
    printf("%d %d %d\n", p[0], p_mappedptr == &p, p0_mappedptr == &p[-1]);
    // CHECK: 222 1 1
    p++;
  }

#pragma omp target map(present, alloc : p) firstprivate(p_mappedptr, p0_mappedptr)
  {
    printf("%d %d %d\n", p[0], p_mappedptr == &p, p0_mappedptr == &p[-2]);
    // CHECK: 333 1 1
  }

#pragma omp target exit data map(delete:p[0 : 5], p)
}

int main() { f1(); }
