// RUN: %libomptarget-compilexx-generic
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic
// RUN: env OMP_TARGET_OFFLOAD=disabled %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

// Test that when a use_device_ptr lookup fails, the
// privatized pointer retains its original value by
// default.
//
// This is necessary because we must assume that the
// pointee is device-accessible, even if it was not
// previously mapped.

#include <stdio.h>
int x;
int *xp = &x;
int *&xpr = xp;

void f2() {
  printf("%p\n", xpr); // CHECK:      0x[[#%x,ADDR:]]
#pragma omp target data use_device_ptr(xpr)
  printf("%p\n", xpr); // CHECK-NEXT: 0x{{0*}}[[#ADDR]]
}

int main() { f2(); }
