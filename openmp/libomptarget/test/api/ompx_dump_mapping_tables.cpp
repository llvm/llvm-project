// RUN: %libomptarget-compilexx-run-and-check-generic

#include <cstdio>
#include <omp.h>

#define N 10

int main() {
  int *a = new __int32_t[N];     // mapped and released from device 0
  int *b = new __int32_t[2 * N]; // mapped to device 0

  // clang-format off
  // CHECK: Mapping tables after target enter data:
  // CHECK-NEXT: omptarget device 0 info: OpenMP Host-Device pointer mappings after block
  // CHECK-NEXT: omptarget device 0 info: Host Ptr Target Ptr Size (B) DynRefCount HoldRefCount Declaration
  // CHECK-NEXT: omptarget device 0 info: {{(0x[0-9a-f]{16})}} {{(0x[0-9a-f]{16})}} {{[48]}}0
  // CHECK-NEXT: omptarget device 0 info: {{(0x[0-9a-f]{16})}} {{(0x[0-9a-f]{16})}} {{[48]}}0
#pragma omp target enter data device(0) map(to : a[ : N])
#pragma omp target enter data device(0) map(to : b[ : 2*N])
  // clang-format on
  printf("Mapping tables after target enter data:\n");
  ompx_dump_mapping_tables();

  // clang-format off
  // CHECK: Mapping tables after target exit data for a:
  // CHECK-NEXT: omptarget device 0 info: OpenMP Host-Device pointer mappings after block
  // CHECK-NEXT: omptarget device 0 info: Host Ptr Target Ptr Size (B) DynRefCount HoldRefCount Declaration
  // CHECK-NEXT: omptarget device 0 info: {{(0x[0-9a-f]{16})}} {{(0x[0-9a-f]{16})}} 80
#pragma omp target exit data device(0) map(release : a[ : N])
  // clang-format on
  printf("\nMapping tables after target exit data for a:\n");
  ompx_dump_mapping_tables();

  return 0;
}
