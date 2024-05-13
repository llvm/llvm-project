// RUN: %libomptarget-compile-generic -fprofile-instr-generate-gpu
// RUN: env LLVM_PROFILE_FILE=llvm.profraw %libomptarget-run-generic 2>&1
// RUN: llvm-profdata show --all-functions --counts \
// RUN:     %target_triple.llvm.profraw | %fcheck-generic \
// RUN:     --check-prefix="LLVM-PGO"

// RUN: %libomptarget-compile-generic -fprofile-generate-gpu
// RUN: env LLVM_PROFILE_FILE=clang.profraw %libomptarget-run-generic 2>&1
// RUN: llvm-profdata show --all-functions --counts \
// RUN:     %target_triple.clang.profraw | %fcheck-generic \
// RUN:     --check-prefix="CLANG-PGO"

// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// REQUIRES: pgo

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

// LLVM-PGO-LABEL: __omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}:
// LLVM-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// LLVM-PGO: Counters: 4
// LLVM-PGO: Function count: 20
// LLVM-PGO: Block counts: [10, 20, 10]

// LLVM-PGO-LABEL: test1:
// LLVM-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// LLVM-PGO: Counters: 1
// LLVM-PGO: Function count: 1
// LLVM-PGO: Block counts: []

// LLVM-PGO-LABEL: test2:
// LLVM-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// LLVM-PGO: Counters: 1
// LLVM-PGO: Function count: 1
// LLVM-PGO: Block counts: []

// CLANG-PGO-LABEL: __omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}:
// CLANG-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// CLANG-PGO: Counters: 3
// CLANG-PGO: Function count: 0
// CLANG-PGO: Block counts: [11, 20]

// CLANG-PGO-LABEL: test1:
// CLANG-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// CLANG-PGO: Counters: 1
// CLANG-PGO: Function count: 10
// CLANG-PGO: Block counts: []

// CLANG-PGO-LABEL: test2:
// CLANG-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// CLANG-PGO: Counters: 1
// CLANG-PGO: Function count: 20
// CLANG-PGO: Block counts: []
