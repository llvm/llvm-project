// RUN: %libomptarget-compilexx-run-and-check-generic

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

void f1() {
  printf("%p\n", xp); // CHECK:      0x[[#%x,ADDR:]]
#pragma omp target data use_device_ptr(xp)
  printf("%p\n", xp); // CHECK-NEXT: 0x{{0*}}[[#ADDR]]
}

int main() { f1(); }
