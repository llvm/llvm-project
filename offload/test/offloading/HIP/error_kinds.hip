// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t
// RUN: %t | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t -fopenmp
// RUN: %t | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO
// UNSUPPORTED: amdgpu-amd-amdhsa-LTO
// UNSUPPORTED: intelgpu

#include <stdio.h>

static void print_error(const char *Label, hipError_t Error) {
  printf("%s value: %u\n", Label, static_cast<unsigned>(Error));
  printf("%s name: %s\n", Label, hipGetErrorName(Error));
  printf("%s string: %s\n", Label, hipGetErrorString(Error));
}

int main() {
  print_error("success", hipSuccess);
  // CHECK: success value: 0
  // CHECK: success name: hipSuccess
  // CHECK: success string: No error

  print_error("invalid value", hipErrorInvalidValue);
  // CHECK: invalid value value: 1
  // CHECK: invalid value name: hipErrorInvalidValue
  // CHECK: invalid value string: Invalid argument value

  print_error("invalid device", hipErrorInvalidDevice);
  // CHECK: invalid device value: 2
  // CHECK: invalid device name: hipErrorInvalidDevice
  // CHECK: invalid device string: Invalid device number

  print_error("unknown", hipErrorUnknown);
  // CHECK: unknown value: 3
  // CHECK: unknown name: Unrecognized error
  // CHECK: unknown string: Unknown error

  hipError_t Unrecognized = static_cast<hipError_t>(999);
  print_error("unrecognized", Unrecognized);
  // CHECK: unrecognized value: 999
  // CHECK: unrecognized name: Unrecognized error
  // CHECK: unrecognized string: Unrecognized error

  print_error("set invalid device", hipSetDevice(-1));
  // CHECK: set invalid device value: 2
  // CHECK: set invalid device name: hipErrorInvalidDevice
  // CHECK: set invalid device string: Invalid device number

  print_error("null stream destroy", hipStreamDestroy(nullptr));
  // CHECK: null stream destroy value: 1
  // CHECK: null stream destroy name: hipErrorInvalidValue
  // CHECK: null stream destroy string: Invalid argument value

  return 0;
}
