// RUN: %libomptarget-compile-generic -fprofile-instr-generate \
// RUN:     -Xclang "-fprofile-instrument=clang"
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic \
// RUN:     --check-prefix="CLANG-PGO"
// RUN: %libomptarget-compile-generic -fprofile-generate \
// RUN:     -Xclang "-fprofile-instrument=llvm"
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic \
// RUN:     --check-prefix="LLVM-PGO"

// REQUIRES: gpu
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

// CLANG-PGO: ======== Counters =========
// CLANG-PGO-NEXT: [ 0 11 20 ]
// CLANG-PGO-NEXT: [ 10 ]
// CLANG-PGO-NEXT: [ 20 ]
// CLANG-PGO-NEXT: ========== Data ===========
// CLANG-PGO-NEXT: { {{[0-9]*}} {{[0-9]*}}
// CLANG-PGO-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// CLANG-PGO-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// CLANG-PGO-SAME: {{[0-9]*}} {{.*}} {{[0-9]*}} }
// CLANG-PGO-NEXT: { {{[0-9]*}} {{[0-9]*}}
// CLANG-PGO-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// CLANG-PGO-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// CLANG-PGO-SAME: {{[0-9]*}} {{.*}} {{[0-9]*}} }
// CLANG-PGO-NEXT: { {{[0-9]*}} {{[0-9]*}}
// CLANG-PGO-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// CLANG-PGO-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// CLANG-PGO-SAME: {{[0-9]*}} {{.*}} {{[0-9]*}} }
// CLANG-PGO-NEXT: ======== Functions ========
// CLANG-PGO-NEXT: pgo1.c:
// CLANG-PGO-SAME: __omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}
// CLANG-PGO-NEXT: test1
// CLANG-PGO-NEXT: test2

// LLVM-PGO: ======== Counters =========
// LLVM-PGO-NEXT: [ 20 10 2 1 ]
// LLVM-PGO-NEXT: [ 10 ]
// LLVM-PGO-NEXT: [ 20 ]
// LLVM-PGO-NEXT: ========== Data ===========
// LLVM-PGO-NEXT: { {{[0-9]*}} {{[0-9]*}}
// LLVM-PGO-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// LLVM-PGO-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// LLVM-PGO-SAME: {{[0-9]*}} {{.*}} {{[0-9]*}} }
// LLVM-PGO-NEXT: { {{[0-9]*}} {{[0-9]*}}
// LLVM-PGO-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// LLVM-PGO-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// LLVM-PGO-SAME: {{[0-9]*}} {{.*}} {{[0-9]*}} }
// LLVM-PGO-NEXT: { {{[0-9]*}} {{[0-9]*}}
// LLVM-PGO-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// LLVM-PGO-SAME: {{0x[0-9a-fA-F]*}} {{0x[0-9a-fA-F]*}}
// LLVM-PGO-SAME: {{[0-9]*}} {{.*}} {{[0-9]*}} }
// LLVM-PGO-NEXT: ======== Functions ========
// LLVM-PGO-NEXT: __omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}
// LLVM-PGO-NEXT: test1
// LLVM-PGO-NEXT: test2
