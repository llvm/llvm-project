// RUN: %libomptarget-compile-generic
//
// RUN: env LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS=1 \
// RUN: env LIBOMPTARGET_DEBUG=1 \
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=DEBUG
//
// RUN: env LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS=1 \
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=CHECK
//
// REQUIRES: debug

// Ensure that under LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS, the pointer
// attachment for map(p[0:0]) happens as if the user had specified the
// attach(always) map-type-modifier.

#include <omp.h>
#include <stdio.h>

int x[10];
int *p;

void f1() {
#pragma omp target enter data map(to : p)
#pragma omp target enter data map(to : x)

  p = &x[0];
  int **p_mappedptr = (int **)omp_get_mapped_ptr(&p, omp_get_default_device());
  int *x0_mappedptr =
      (int *)omp_get_mapped_ptr(&x[0], omp_get_default_device());
  int *p0_deviceaddr = NULL;

  printf("p_mappedptr %s null\n", p_mappedptr == (int **)NULL ? "==" : "!=");
  printf("x0_mappedptr %s null\n", x0_mappedptr == (int *)NULL ? "==" : "!=");
  // CHECK: p_mappedptr != null
  // CHECK: x0_mappedptr != null

#pragma omp target enter data map(to : p[0 : 0]) // Implies: attach(auto)
  // clang-format off
  // DEBUG: omptarget --> Treating ATTACH(auto) as ATTACH(always) because LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS is true
  // clang-format on

#pragma omp target map(present, alloc : p) map(from : p0_deviceaddr)
  {
    p0_deviceaddr = &p[0];
  }

  printf("p0_deviceaddr %s x0_mappedptr\n",
         p0_deviceaddr == x0_mappedptr ? "==" : "!=");
  // CHECK: p0_deviceaddr == x0_mappedptr

#pragma omp target exit data map(delete : x, p)
}

int main() { f1(); }
