// RUN: %libomptarget-compilexx-run-and-check-generic

// REQUIRES: libc

#include <omp.h>
#include <stdio.h>

int x[10];

void f1() {
  int *p;
  p = &x[0];
  p[0] = 111;
  p[1] = 222;
  p[2] = 333;
  p[3] = 444;

#pragma omp target enter data map(to : p)
#pragma omp target enter data map(to : p[0 : 5])

  int **p_mappedptr = (int **)omp_get_mapped_ptr(&p, omp_get_default_device());
  int *x0_mappedptr =
      (int *)omp_get_mapped_ptr(&x[0], omp_get_default_device());
  int *x0_hostaddr = &x[0];

  printf("p_mappedptr %s null\n", p_mappedptr == (int **)NULL ? "==" : "!=");
  printf("x0_mappedptr %s null\n", x0_mappedptr == (int *)NULL ? "==" : "!=");

// CHECK: p_mappedptr != null
// CHECK: x0_mappedptr != null

// p is predetermined firstprivate, so its address will be different from
// the mapped address for this construct. So, any changes to p within the
// region will not be visible after the construct.
#pragma omp target map(p[0]) map(to : p_mappedptr, x0_mappedptr, x0_hostaddr)
  {
    printf("%d %d %d %d\n", p[0], p_mappedptr == &p, x0_mappedptr == &p[0],
           x0_hostaddr == &p[0]);
    // CHECK:    111 0 1 0
    p++;
  }

// For the remaining constructs, p is not firstprivate, so its address will
// be the same as the mapped address, and changes to p will be visible to any
// subsequent regions.
#pragma omp target map(to : p[0], p)                                           \
    map(to : p_mappedptr, x0_mappedptr, x0_hostaddr)
  {
    printf("%d %d %d %d\n", p[0], p_mappedptr == &p, x0_mappedptr == &p[0],
           x0_hostaddr == &p[0]);
    // EXPECTED: 111 1 1 0
    // CHECK:    111 0 1 0
    p++;
  }

#pragma omp target map(to : p, p[0])                                           \
    map(to : p_mappedptr, x0_mappedptr, x0_hostaddr)
  {
    printf("%d %d %d %d\n", p[0], p_mappedptr == &p, x0_mappedptr == &p[-1],
           x0_hostaddr == &p[-1]);
    // EXPECTED: 222 1 1 0
    // CHECK:    111 0 0 0
    p++;
  }

#pragma omp target map(present, alloc : p)                                     \
    map(to : p_mappedptr, x0_mappedptr, x0_hostaddr)
  {
    printf("%d %d %d %d\n", p[0], p_mappedptr == &p, x0_mappedptr == &p[-2],
           x0_hostaddr == &p[-2]);
    // EXPECTED: 333 1 1 0
    // CHECK:    111 1 0 0
  }

  // The following map(from:p) should not bring back p, because p is an
  // attached pointer. So, it should still point to the same original
  // location, &x[0], on host.
#pragma omp target exit data map(always, from : p)
  printf("%d %d\n", p[0], p == &x[0]);
  // CHECK: 111 1

#pragma omp target exit data map(delete : p[0 : 5], p)
}

int main() { f1(); }
