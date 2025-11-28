// RUN: %libomptarget-compilexx-run-and-check-generic

// Test that when a use_device_ptr lookup fails, the
// privatized pointer retains its original value by
// default.
//
// This is necessary because we must assume that the
// pointee is device-accessible, even if it was not
// previously mapped.
//
// OpenMP 5.1, sec 2.14.2, target data construct, p 188, l26-31:
// If a list item that appears in a use_device_ptr clause ... does not point to
// a mapped object, it must contain a valid device address for the target
// device, and the list item references are instead converted to references to a
// local device pointer that refers to this device address.
//
// Note: OpenMP 6.1 will have a way to change the
// fallback behavior: preserve or nullify.

// XFAIL: *

#include <stdio.h>
int x;
int *xp = &x;

void f1() {
  printf("%p\n", xp); // CHECK:      0x[[#%x,ADDR:]]
#pragma omp target data use_device_ptr(xp)
  printf("%p\n", xp); // CHECK-NEXT: 0x{{0*}}[[#ADDR]]
}

int main() { f1(); }
