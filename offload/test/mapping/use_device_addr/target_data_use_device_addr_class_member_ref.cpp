// RUN: %libomptarget-compilexx-run-and-check-generic

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

  void f8() {
#pragma omp target enter data map(to : d)
    {
      void *mapped_ptr = omp_get_mapped_ptr(&d, omp_get_default_device());
      printf("%d\n", mapped_ptr != NULL); // CHECK: 1
#pragma omp target data use_device_addr(d)
      {
        printf("%d\n", &d == mapped_ptr); // CHECK: 1
      }
    }
  }
};

int main() {
  ST s;
  s.f8();
}
