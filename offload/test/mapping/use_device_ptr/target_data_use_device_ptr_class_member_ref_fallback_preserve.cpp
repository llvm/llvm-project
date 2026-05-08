// RUN: %libomptarget-compilexx-generic -fopenmp-version=61
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic
// RUN: env OMP_TARGET_OFFLOAD=disabled %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// Test that when a use_device_ptr lookup fails, the
// privatized pointer retains its original value
// because of fb_preserve.

#include <stdio.h>

int x = 0;
int *y = &x;

struct ST {
  int *&b = y;

  void f2() {
    printf("%p\n", b); // CHECK:      0x[[#%x,ADDR:]]
#pragma omp target data use_device_ptr(fb_preserve : b)
    printf("%p\n", b); // CHECK-NEXT: 0x{{0*}}[[#ADDR]]
  }
};

int main() {
  ST s;
  s.f2();
}
