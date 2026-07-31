// Pointer + pointee mapping with the `close` map-type modifier under unified
// shared memory, where the pointer `p` and its pointee `p[0:10]` are mapped on
// the SAME `target data` directive.
//
// Companion to close_ptr_ptee_nested.c; see that file for the rationale. Here
// the pointer and pointee are processed together in a single targetDataBegin,
// so pointer-attachment sees both the pointer and the (close-allocated) pointee
// in the same construct.
//
// Each variation is run twice: once under LIBOMPTARGET_DEBUG=1 to check the
// pointer-attachment bookkeeping, and once to check the program results.
//
// RUN: %libomptarget-compile-generic -DV1=1
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefixes=ATTACHED
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefixes=V1,ALL
//
// RUN: %libomptarget-compile-generic -DV2=1
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefixes=ATTACHED
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefixes=V2,ALL
//
// RUN: %libomptarget-compile-generic
// RUN: env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefixes=ATTACHED
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefixes=V3,ALL
//
// REQUIRES: unified_shared_memory
// REQUIRES: libomptarget-debug
//
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

int x[10] = {0};
int *p = &x[0];
int *p_device = NULL;
int **paddr_device = NULL;

int main() {
  printf("Before tgt: p %s p_host\n", p == &x[0] ? "==" : "!=");

#if V1
#pragma omp target data map(tofrom : p) map(close, tofrom : p[0 : 10])
#elif V2
#pragma omp target data map(close, tofrom : p) map(tofrom : p[0 : 10])
#else // V3
#pragma omp target data map(close, tofrom : p) map(close, tofrom : p[0 : 10])
#endif
  {
#pragma omp target map(present, alloc : p) map(from : p_device, paddr_device)
    {
      p_device = p;
      paddr_device = &p;
      p[0] = 111;
    }
  }

  printf("In tgt: p_device %s p_host\n", p_device == &x[0] ? "==" : "!=");
  printf("In tgt: paddr_device %s &p_host\n", paddr_device == &p ? "==" : "!=");
  printf("After tgt: p %s p_host\n", p == &x[0] ? "==" : "!=");
  printf("p[0] = %d\n", x[0]);

  // When the pointee has a device allocation of its own, the pointer is attached
  // to it and its original host value is restored at the end of the region.
  // ATTACHED: ATTACH entry {{.*}} processed successfully
  // ATTACHED: Restoring host pointer


  // ALL: Before tgt: p == p_host

  // The close pointee is device-allocated and p (host path) is attached to it.
  // V1: In tgt: p_device != p_host
  // V1: In tgt: paddr_device == &p_host

  // p is close-allocated (own device storage, so &p differs); the pointee is a
  // plain USM mapping, so the device pointer value equals the host address.
  // V2: In tgt: p_device == p_host
  // V2: In tgt: paddr_device != &p_host

  // Both p and the pointee are close-allocated, so both the device address of p
  // and its attached value differ from the host.
  // V3: In tgt: p_device != p_host
  // V3: In tgt: paddr_device != &p_host

  // ALL: After tgt: p == p_host
  // ALL: p[0] = 111
}
