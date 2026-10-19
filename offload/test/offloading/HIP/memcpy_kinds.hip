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

int main(int argc, char **argv) {
  int HostSrc = 11;
  int HostDst = 0;
  if (hipMemcpy(&HostDst, &HostSrc, sizeof(int), hipMemcpyHostToHost) !=
      hipSuccess)
    return 1;

  printf("host to host: %d\n", HostDst);
  // CHECK: host to host: 11

  int *DevSrc = nullptr;
  int *DevDst = nullptr;
  int Result = 0;
  if (hipMalloc(&DevSrc, sizeof(int)) != hipSuccess)
    return 1;
  if (hipMalloc(&DevDst, sizeof(int)) != hipSuccess)
    return 1;

  HostSrc = 42;
  if (hipMemcpy(DevSrc, &HostSrc, sizeof(int), hipMemcpyHostToDevice) !=
      hipSuccess)
    return 1;
  if (hipMemcpy(DevDst, DevSrc, sizeof(int), hipMemcpyDeviceToDevice) !=
      hipSuccess)
    return 1;
  if (hipMemcpy(&Result, DevDst, sizeof(int), hipMemcpyDeviceToHost) !=
      hipSuccess)
    return 1;

  printf("device to device: %d\n", Result);
  // CHECK: device to device: 42

  hipFree(DevSrc);
  hipFree(DevDst);
}
