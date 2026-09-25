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
int test2(int a) { return a * 2; }

int main() {
  int device_var = 1;

#pragma omp target teams distribute parallel for num_teams(3)                  \
    map(tofrom : device_var)
  for (int i = 1; i <= 30; i++) {
    device_var *= i;
    if (i % 2 == 0) {
      device_var += test1(device_var);
    }
    if (i % 3 == 0) {
      device_var += test2(device_var);
    }
  }
}

// clang-format off
// LLVM-PGO-DAG: {{(\[| )}}30{{(,|\])}}
// LLVM-PGO-DAG: {{(\[| )}}15{{(,|\])}}
// LLVM-PGO-DAG: {{(\[| )}}10{{(,|\])}}

// LLVM-PGO-LABEL: test1:
// LLVM-PGO: Block counts: [15]

// LLVM-PGO-LABEL: test2:
// LLVM-PGO: Block counts: [10]

// LLVM-PGO-LABEL: Instrumentation level:
// LLVM-PGO-SAME: IR

// CLANG-PGO-DAG: {{(\[| )}}15{{(,|\])}}
// CLANG-PGO-DAG: {{(\[| )}}10{{(,|\])}}

// CLANG-PGO-LABEL: test1:
// CLANG-PGO: Function count: 15

// CLANG-PGO-LABEL: test2:
// CLANG-PGO: Function count: 10

// CLANG-PGO-LABEL: Instrumentation level:
// CLANG-PGO-SAME: Front-end
// clang-format on
