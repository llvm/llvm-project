// RUN: %libomptarget-compile-generic -fcreate-profile \
// RUN:     -Xarch_device -fprofile-generate
// RUN: env LLVM_PROFILE_FILE=%basename_t.llvm.profraw \
// RUN:     %libomptarget-run-generic 2>&1
// RUN: %profdata show --all-functions --counts \
// RUN:     %target_triple.%basename_t.llvm.profraw | \
// RUN:     %fcheck-generic --check-prefix="LLVM-PGO"

// RUN: %libomptarget-compile-generic -fcreate-profile \
// RUN:     -Xarch_device -fprofile-instr-generate
// RUN: env LLVM_PROFILE_FILE=%basename_t.clang.profraw \
// RUN:     %libomptarget-run-generic 2>&1
// RUN: %profdata show --all-functions --counts \
// RUN:     %target_triple.%basename_t.clang.profraw | \
// RUN:     %fcheck-generic --check-prefix="CLANG-PGO"

// REQUIRES: amdgpu
// REQUIRES: pgo
// XFAIL: amdgpu

int test1(int a) { return a / 2; }
int test2(int a) { return a * 2; }

int main() {
  int m = 2;
#pragma omp target
  {
    for (int i = 0; i < 10; i++) {
      m = test1(m);
      for (int j = 0; j < 2; j++) {
        m = test2(m);
      }
    }
  }
}

// LLVM-PGO-LABEL: __omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}:
// LLVM-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// LLVM-PGO: Counters: 4
// LLVM-PGO: Block counts: [20, 10, {{.*}}, 1]

// LLVM-PGO-LABEL: test1:
// LLVM-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// LLVM-PGO: Counters: 1
// LLVM-PGO: Block counts: [10]

// LLVM-PGO-LABEL: test2:
// LLVM-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// LLVM-PGO: Counters: 1
// LLVM-PGO: Block counts: [20]

// LLVM-PGO-LABEL: Instrumentation level:
// LLVM-PGO-SAME: IR
// LLVM-PGO-SAME: entry_first = 0
// LLVM-PGO-LABEL: Functions shown:
// LLVM-PGO-SAME: 3
// LLVM-PGO-LABEL: Maximum function count:
// LLVM-PGO-SAME: 20

// CLANG-PGO-LABEL: __omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}:
// CLANG-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// CLANG-PGO: Block counts: [10, 20]

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

// CLANG-PGO-LABEL: Instrumentation level:
// CLANG-PGO-SAME: Front-end
// CLANG-PGO-LABEL: Functions shown:
// CLANG-PGO-SAME: 3
// CLANG-PGO-LABEL: Maximum internal block count:
// CLANG-PGO-SAME: 20
