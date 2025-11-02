// RUN: %libomptarget-compile-generic -fprofile-generate
// RUN: env LLVM_PROFILE_FILE=%basename_t.llvm.profraw \
// RUN:     %libomptarget-run-generic 2>&1
// RUN: %profdata show --all-functions --counts \
// RUN:     %basename_t.llvm.profraw | %fcheck-generic \
// RUN:     --check-prefix="LLVM-HOST"
// RUN: %profdata show --all-functions --counts \
// RUN:     %target_triple.%basename_t.llvm.profraw \
// RUN:     | %fcheck-generic --check-prefix="LLVM-DEVICE"

// RUN: %libomptarget-compile-generic -fprofile-instr-generate
// RUN: env LLVM_PROFILE_FILE=%basename_t.clang.profraw \
// RUN:     %libomptarget-run-generic 2>&1
// RUN: %profdata show --all-functions --counts \
// RUN:     %basename_t.clang.profraw | %fcheck-generic \
// RUN:     --check-prefix="CLANG-HOST"
// RUN: %profdata show --all-functions --counts \
// RUN:     %target_triple.%basename_t.clang.profraw | \
// RUN:     %fcheck-generic --check-prefix="CLANG-DEV"

// RUN: %libomptarget-compile-generic -Xarch_host -fprofile-generate
// RUN: env LLVM_PROFILE_FILE=%basename_t.nogpu.profraw \
// RUN:     %libomptarget-run-generic 2>&1
// RUN: %profdata show --all-functions --counts \
// RUN:     %basename_t.nogpu.profraw | %fcheck-generic \
// RUN:     --check-prefix="LLVM-HOST"
// RUN: not test -e %target_triple.%basename_t.nogpu.profraw

// RUN: %libomptarget-compile-generic -Xarch_host -fprofile-generate \
// RUN:     -Xarch_device -fprofile-instr-generate
// RUN: env LLVM_PROFILE_FILE=%basename_t.hidf.profraw \
// RUN:     %libomptarget-run-generic 2>&1
// RUN: %profdata show --all-functions --counts \
// RUN:     %basename_t.hidf.profraw | %fcheck-generic \
// RUN:     --check-prefix="LLVM-HOST"
// RUN: %profdata show --all-functions --counts \
// RUN:     %target_triple.%basename_t.hidf.profraw \
// RUN:     | %fcheck-generic --check-prefix="CLANG-DEV"

// RUN: %libomptarget-compile-generic -Xarch_device -fprofile-generate \
// RUN:     -Xarch_host -fprofile-instr-generate
// RUN: env LLVM_PROFILE_FILE=%basename_t.hfdi.profraw \
// RUN:     %libomptarget-run-generic 2>&1
// RUN: %profdata show --all-functions --counts \
// RUN:     %basename_t.hfdi.profraw | %fcheck-generic \
// RUN:     --check-prefix="CLANG-HOST"
// RUN: %profdata show --all-functions --counts \
// RUN:     %target_triple.%basename_t.hfdi.profraw \
// RUN:     | %fcheck-generic --check-prefix="LLVM-DEVICE"

// REQUIRES: amdgpu
// REQUIRES: pgo
// XFAIL: amdgpu

int main() {
  int host_var = 0;
  for (int i = 0; i < 20; i++) {
    host_var += i;
  }

  int device_var = 1;
#pragma omp target
  {
    for (int i = 0; i < 10; i++) {
      device_var *= i;
    }
  }
}

// LLVM-HOST-LABEL: main:
// LLVM-HOST: Hash: {{0[xX][0-9a-fA-F]+}}
// LLVM-HOST: Counters: 3
// LLVM-HOST: Block counts: [20, 1, 0]

// LLVM-HOST-LABEL: __omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}:
// LLVM-HOST: Hash: {{0[xX][0-9a-fA-F]+}}
// LLVM-HOST: Counters: 2
// LLVM-HOST: Block counts: [0, 0]
// LLVM-HOST: Instrumentation level: IR

// LLVM-DEVICE-LABEL: __omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}:
// LLVM-DEVICE: Hash: {{0[xX][0-9a-fA-F]+}}
// LLVM-DEVICE: Counters: 3
// LLVM-DEVICE: Block counts: [10, {{.*}}, 1]
// LLVM-DEVICE: Instrumentation level: IR

// CLANG-HOST-LABEL: main:
// CLANG-HOST: Hash: {{0[xX][0-9a-fA-F]+}}
// CLANG-HOST: Counters: 2
// CLANG-HOST: Function count: 1
// CLANG-HOST: Block counts: [20]

// CLANG-HOST-LABEL: __omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}:
// CLANG-HOST: Hash: {{0[xX][0-9a-fA-F]+}}
// CLANG-HOST: Counters: 2
// CLANG-HOST: Function count: 0
// CLANG-HOST: Block counts: [0]
// CLANG-HOST: Instrumentation level: Front-end

// CLANG-DEV-LABEL: __omp_offloading_{{[_0-9a-zA-Z]*}}_main_{{[_0-9a-zA-Z]*}}:
// CLANG-DEV: Hash: {{0[xX][0-9a-fA-F]+}}
// CLANG-DEV: Counters: 2
// CLANG-DEV: Block counts: [10]
// CLANG-DEV: Instrumentation level: Front-end
