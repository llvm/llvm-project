// RUN: %libomptarget-compile-generic -fcreate-profile \
// RUN:     -Xarch_device -fprofile-generate \
// RUN:     -Xarch_device -fprofile-update=atomic
// RUN: env LLVM_PROFILE_FILE=%basename_t.llvm.profraw \
// RUN:     %libomptarget-run-generic 2>&1
// RUN: %profdata show --all-functions --counts \
// RUN:     %target_triple.%basename_t.llvm.profraw | \
// RUN:     %fcheck-generic --check-prefix="LLVM-PGO"

// RUN: %libomptarget-compile-generic -fcreate-profile \
// RUN:     -Xarch_device -fprofile-instr-generate \
// RUN:     -Xarch_device -fprofile-update=atomic
// RUN: env LLVM_PROFILE_FILE=%basename_t.clang.profraw \
// RUN:     %libomptarget-run-generic 2>&1
// RUN: %profdata show --all-functions --counts \
// RUN:     %target_triple.%basename_t.clang.profraw | \
// RUN:     %fcheck-generic --check-prefix="CLANG-PGO"

// REQUIRES: amdgpu
// REQUIRES: pgo

int test1(int a) { return a / 2; }

int main() {
  int device_var = 1;
#pragma omp target map(tofrom : device_var)
  {
#pragma omp parallel for
    for (int i = 1; i <= 10; i++) {
      device_var *= i;
      if (i % 2 == 0) {
        device_var += test1(device_var);
      }
    }
  }
}

// clang-format off
// LLVM-PGO-LABEL: __omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}:
// LLVM-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// LLVM-PGO: Counters: 2
// LLVM-PGO: Block counts: [0, {{.*}}]

// LLVM-PGO-LABEL: __omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}_omp_outlined:
// LLVM-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// LLVM-PGO: Counters: 5
// LLVM-PGO: Block counts: [10, 5, {{.*}}, 10, {{.*}}]

// LLVM-PGO-LABEL: test1:
// LLVM-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// LLVM-PGO: Counters: 1
// LLVM-PGO: Block counts: [5]

// LLVM-PGO-LABEL: Instrumentation level:
// LLVM-PGO-SAME: IR
// LLVM-PGO-SAME: entry_first = 0
// LLVM-PGO-LABEL: Functions shown:
// LLVM-PGO-SAME: 3
// LLVM-PGO-LABEL: Maximum function count:
// LLVM-PGO-SAME: 10

// CLANG-PGO-LABEL: __omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}:
// CLANG-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// CLANG-PGO: Counters: 1
// CLANG-PGO: Function count: {{.*}}
// CLANG-PGO: Block counts: []

// CLANG-PGO-LABEL: __omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}_omp_outlined:
// CLANG-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// CLANG-PGO: Counters: 3
// CLANG-PGO: Function count: {{.*}}
// CLANG-PGO: Block counts: [{{.*}}, 5]

// CLANG-PGO-LABEL: test1:
// CLANG-PGO: Hash: {{0[xX][0-9a-fA-F]+}}
// CLANG-PGO: Counters: 1
// CLANG-PGO: Function count: 5
// CLANG-PGO: Block counts: []

// CLANG-PGO-LABEL: Instrumentation level:
// CLANG-PGO-SAME: Front-end
// CLANG-PGO-LABEL: Functions shown:
// CLANG-PGO-SAME: 3
// clang-format on
