// RUN: %libomptarget-compile-generic -fprofile-instr-generate \
// RUN:     -Xclang "-fprofile-instrument=clang"
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic

// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// REQUIRES: pgo

#ifdef _OPENMP
#include <omp.h>
#endif

int test1(int a) { return a / 2; }
int test2(int a) { return a * 2; }

int main() {
  int m = 2;
#pragma omp target
  for (int i = 0; i < 10; i++) {
    m = test1(m);
    for (int j = 0; j < 2; j++) {
      m = test2(m);
    }
  }
}

// CHECK: ======== Counters =========
// CHECK-NEXT: [ 0 11 20 ]
// CHECK-NEXT: [ 10 ]
// CHECK-NEXT: [ 20 ]
// CHECK-NEXT: ========== Data ===========
// CHECK-NEXT: { {{[0-9]*}} {{[0-9]*}}
// CHECK-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// CHECK-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// CHECK-SAME: {{[0-9]*}} {{[0-9]*}} {{[0-9]*}} }
// CHECK-NEXT: { {{[0-9]*}} {{[0-9]*}}
// CHECK-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// CHECK-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// CHECK-SAME: {{[0-9]*}} {{[0-9]*}} {{[0-9]*}} }
// CHECK-NEXT: { {{[0-9]*}} {{[0-9]*}}
// CHECK-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// CHECK-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// CHECK-SAME: {{[0-9]*}} {{[0-9]*}} {{[0-9]*}} }
// CHECK-NEXT: ======== Functions ========
// CHECK-NEXT: pgo1.c:__omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}
// CHECK-NEXT: test1
// CHECK-NEXT: test2
