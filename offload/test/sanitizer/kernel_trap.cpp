
// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,TRACE,NDEBG 
// RUN: %not --crash %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK
// RUN: %libomptarget-compilexx-generic -g
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,TRACE,DEBUG
// RUN: %not --crash %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK
// clang-format on

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

struct S {};

template <typename T> void cxx_function_name(int I, T *) {

#pragma omp target
  {
  }
#pragma omp target
  {
  }
#pragma omp target
  {
    __builtin_trap();
  }
#pragma omp target
  {
  }
}

int main(void) {
  struct S s;
  cxx_function_name(1, &s);
}

// clang-format off
// CHECK: OFFLOAD ERROR: Kernel 'omp target in void cxx_function_name<S>(int, S*) @ [[LINE:[0-9]+]] (__omp_offloading_{{.*}}__Z17cxx_function_nameI1SEviPT__l[[LINE]])'
// CHECK: OFFLOAD ERROR: execution interrupted by hardware trap instruction
// TRACE:     launchKernel
// NDEBG:     cxx_function_name<S>(int, S*)
// NDEBG:     main
// DEBUG:     cxx_function_name<S>(int, S*) {{.*}}kernel_trap.cpp:
// DEBUG:     main {{.*}}kernel_trap.cpp:
// clang-format on
