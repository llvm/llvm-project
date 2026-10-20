// RUN: %libomptarget-compilexx-generic -fopenmp-version=61
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefixes=CHECK,OFFLOAD
// RUN: env OMP_TARGET_OFFLOAD=disabled %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefixes=CHECK,NOOFFLOAD

// Test that when a use_device_ptr lookup fails, the
// privatized pointer is set to null because of fb_nullify.

#include <stdio.h>

int x = 0;
int *y = &x;

struct ST {
  int *&b = y;

  void f2() {
    printf("%p\n", b); // CHECK:          0x[[#%x,ADDR:]]
#pragma omp target data use_device_ptr(fb_nullify : b)
    printf("%p\n", b); // OFFLOAD-NEXT:   (nil)
                       // NOOFFLOAD-NEXT: 0x{{0*}}[[#ADDR]]
  }
};

int main() {
  ST s;
  s.f2();
}
