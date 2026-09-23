// RUN: %libomptarget-compilexx-run-and-check-generic

// Test that when a use_device_addr lookup fails, the
// list-item retains its original address by default.
//
// This is necessary because we must assume that the
// list-item is device-accessible, even if it was not
// previously mapped.

#include <stdio.h>
int h[10];
int *ph = &h[0];

void f1() {
  printf("%p\n", &h[2]); // CHECK:      0x[[#%x,ADDR:]]
#pragma omp target data use_device_addr(h[2])
  printf("%p\n", &h[2]); // CHECK-NEXT: 0x{{0*}}[[#ADDR]]
#pragma omp target data use_device_addr(ph[2])
  printf("%p\n", &ph[2]); // CHECK-NEXT: 0x{{0*}}[[#ADDR]]
}

int main() { f1(); }
