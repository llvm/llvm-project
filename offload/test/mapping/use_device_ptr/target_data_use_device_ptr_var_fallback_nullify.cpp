// RUN: %libomptarget-compilexx-generic -fopenmp-version=61
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefixes=CHECK,OFFLOAD
// RUN: env OMP_TARGET_OFFLOAD=disabled %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefixes=CHECK,NOOFFLOAD

// Test that when a use_device_ptr lookup fails, the
// privatized pointer is set to null because of fb_nullify.

#include <stdio.h>
int x;
int *xp = &x;

void f1() {
  printf("%p\n", xp); // CHECK:          0x[[#%x,ADDR:]]
#pragma omp target data use_device_ptr(fb_nullify : xp)
  printf("%p\n", xp); // OFFLOAD-NEXT:   (nil)
                      // NOOFFLOAD-NEXT: 0x{{0*}}[[#ADDR]]
}

int main() { f1(); }
