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
int *&xpr = xp;

void f2() {
  printf("%p\n", xpr); // CHECK:          0x[[#%x,ADDR:]]
  // FIXME: We won't get "nil" until we start privatizing xpr.
#pragma omp target data use_device_ptr(fb_nullify : xpr)
  printf("%p\n", xpr); // EXPECTED-OFFLOAD-NEXT:   (nil)
                       // OFFLOAD-NEXT: 0x{{0*}}[[#ADDR]]
                       // NOOFFLOAD-NEXT: 0x{{0*}}[[#ADDR]]
}

int main() { f2(); }
