// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda
// RUN: %t.cuda | %fcheck-generic --check-prefixes=CHECK,CUDA
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda.omp -fopenmp
// RUN: %t.cuda.omp | %fcheck-generic --check-prefixes=CHECK,CUDA
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip
// RUN: %t.hip | %fcheck-generic --check-prefixes=CHECK,HIP
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip.omp -fopenmp
// RUN: %t.hip.omp | %fcheck-generic --check-prefixes=CHECK,HIP
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO
// UNSUPPORTED: amdgpu-amd-amdhsa-LTO
// UNSUPPORTED: intelgpu

// clang-format off
#include <stdio.h>
#include "Inputs/DefineTestLanguageNames.inc"
// clang-format on

static void print_error(const char *Label, Error_t Error) {
  printf("%s value: %u\n", Label, static_cast<unsigned>(Error));
  printf("%s name: %s\n", Label, GetErrorName(Error));
  printf("%s string: %s\n", Label, GetErrorString(Error));
}

int main() {
  print_error("success", Success);
  // CHECK: success value: 0
  // CUDA: success name: cudaSuccess
  // HIP: success name: hipSuccess
  // CHECK: success string: No error

  print_error("invalid value", ErrorInvalidValue);
  // CHECK: invalid value value: 1
  // CUDA: invalid value name: cudaErrorInvalidValue
  // HIP: invalid value name: hipErrorInvalidValue
  // CHECK: invalid value string: Invalid argument value

  print_error("invalid device", ErrorInvalidDevice);
  // CHECK: invalid device value: 2
  // CUDA: invalid device name: cudaErrorInvalidDevice
  // HIP: invalid device name: hipErrorInvalidDevice
  // CHECK: invalid device string: Invalid device number

  print_error("unknown", ErrorUnknown);
  // CHECK: unknown value: 3
  // CHECK: unknown name: Unrecognized error
  // CHECK: unknown string: Unknown error

  print_error("invalid resource handle", ErrorInvalidResourceHandle);
  // CHECK: invalid resource handle value: 4
  // CUDA: invalid resource handle name: cudaErrorInvalidResourceHandle
  // HIP: invalid resource handle name: hipErrorInvalidResourceHandle
  // CHECK: invalid resource handle string: Invalid resource handle

  print_error("invalid configuration", ErrorInvalidConfiguration);
  // CHECK: invalid configuration value: 5
  // CUDA: invalid configuration name: cudaErrorInvalidConfiguration
  // HIP: invalid configuration name: hipErrorInvalidConfiguration
  // CHECK: invalid configuration string: Invalid configuration argument

  Error_t Unrecognized = static_cast<Error_t>(999);
  print_error("unrecognized", Unrecognized);
  // CHECK: unrecognized value: 999
  // CHECK: unrecognized name: Unrecognized error
  // CHECK: unrecognized string: Unrecognized error

  print_error("set invalid device", SetDevice(-1));
  // CHECK: set invalid device value: 2
  // CUDA: set invalid device name: cudaErrorInvalidDevice
  // HIP: set invalid device name: hipErrorInvalidDevice
  // CHECK: set invalid device string: Invalid device number

  print_error("get last error", GetLastError());
  // CHECK: get last error value: 2
  // CUDA: get last error name: cudaErrorInvalidDevice
  // HIP: get last error name: hipErrorInvalidDevice
  // CHECK: get last error string: Invalid device number

  print_error("cleared last error", GetLastError());
  // CHECK: cleared last error value: 0
  // CUDA: cleared last error name: cudaSuccess
  // HIP: cleared last error name: hipSuccess
  // CHECK: cleared last error string: No error

  print_error("null stream destroy", StreamDestroy(nullptr));
  // CHECK: null stream destroy value: 1
  // CUDA: null stream destroy name: cudaErrorInvalidValue
  // HIP: null stream destroy name: hipErrorInvalidValue
  // CHECK: null stream destroy string: Invalid argument value

  return 0;
}
