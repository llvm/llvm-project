// clang-format off
// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: not %libomptarget-run-generic 2>&1 > %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

// Align lines.

#include <stdint.h>
#include <stdio.h>

int main(void) {

  void *Null = 0;
  void *Heap, *Stack;
#pragma omp target map(from : Heap, Stack)
  {
    int Q[512];
    // clang-format off
    // CHECK:      ERROR: OffloadSanitizer out-of-bounds access on address 0x0000000000000000 at pc [[PC:0x.*]]
    // CHECK-NEXT: WRITE of size 4 at 0x0000000000000000 thread <0, 0, 0> block <0, 0, 0>
    // CHECK-NEXT: #0 [[PC]] main null.c:[[@LINE+3]]
    // CHECK-NEXT: 0x0000000000000000 is located 0 bytes inside of 0-byte region [0x0000000000000000,0x0000000000000000)
    // clang-format on
    //    *Null = 42;
    Stack = &Q[0];
    Heap = Null;
  }
  printf("Heap %p Stack %p\n", Heap, Stack);
  printf("Heap %lu Stack %lu\n", ((uintptr_t)Heap & (1UL << 63)),
         ((uintptr_t)Stack & (1UL << 63)));
}
