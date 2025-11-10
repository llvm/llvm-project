// RUN: %libomptarget-compilexx-run-and-check-generic

// XFAIL: *

#include <omp.h>
#include <stdio.h>

int x = 0;
int *y = &x;
int z = 0;

struct ST {
  int n = 111;
  int *a = &x;
  int *&b = y;
  int c = 0;
  int &d = z;
  int m = 0;

  void f2() {
#pragma omp target data map(to : b[0])
    {
      void *mapped_ptr = omp_get_mapped_ptr(b, omp_get_default_device());
      printf("%d\n", mapped_ptr != NULL); // CHECK: 1
#pragma omp target data map(b[0], m) use_device_ptr(b)
      {
        printf("%d\n", b == mapped_ptr); // CHECK: 1
      }
    }
  }
};

int main() {
  ST s;
  s.f2();
}
