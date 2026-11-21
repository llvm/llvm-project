// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda
// RUN: %t.cuda | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda.omp -fopenmp
// RUN: %t.cuda.omp | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip
// RUN: %t.hip | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip.omp -fopenmp
// RUN: %t.hip.omp | %fcheck-generic
// clang-format on

// REQUIRES: gpu, libc

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

__global__ void math_kernel(double *D, float *F) {
  D[0] = sqrt(D[0]);
  D[1] = atan(D[1]);
  D[2] = cos(D[2]);
  D[3] = sin(D[3]);
  D[4] = fabs(D[4]);
  D[5] = floor(D[5]);
  D[6] = ceil(D[6]);
  D[7] = pow(D[7], D[8]);
  D[8] = fma(D[8], 2.0, 1.0);
  D[9] = copysign(D[9], D[0]);
  D[10] = fmod(D[10], D[11]);
  D[11] = exp2(D[12]);
  D[12] = log2(D[11]);
  D[13] = trunc(D[13]);
  D[14] = round(D[14]);
  D[15] = tan(D[15]);

  F[0] = sqrtf(F[0]);
  F[1] = atanf(F[1]);
  F[2] = cosf(F[2]);
  F[3] = sinf(F[3]);
  F[4] = fabsf(F[4]);
  F[5] = floorf(F[5]);
  F[6] = ceilf(F[6]);
  F[7] = powf(F[7], F[8]);
  F[8] = fmaf(F[8], 2.0f, 1.0f);
  F[9] = copysignf(F[9], F[0]);
  F[10] = fmodf(F[10], F[11]);
  F[11] = exp2f(F[12]);
  F[12] = log2f(F[11]);
  F[13] = truncf(F[13]);
  F[14] = roundf(F[14]);
  F[15] = tanf(F[15]);
}

int main(int argc, char **argv) {
  double DHost[16] = {9.0, 0.0,  0.0, 0.0, -5.0, 1.25, 1.75, 2.0,
                      3.0, -2.0, 5.0, 2.0, 1.0,  2.7,  2.4,  0.0};
  float FHost[16] = {16.0f, 0.0f,  0.0f, 0.0f, -5.0f, 1.25f, 1.75f, 2.0f,
                     3.0f,  -2.0f, 5.0f, 2.0f, 1.0f,  2.7f,  2.4f,  0.0f};

  double *D = nullptr;
  float *F = nullptr;

  if (Malloc(&D, sizeof(DHost)) != Success)
    return 1;
  if (Malloc(&F, sizeof(FHost)) != Success)
    return 1;

  if (Memcpy(D, DHost, sizeof(DHost), MemcpyHostToDevice) != Success)
    return 1;
  if (Memcpy(F, FHost, sizeof(FHost), MemcpyHostToDevice) != Success)
    return 1;

  math_kernel<<<1, 1>>>(D, F);
  if (DeviceSynchronize() != Success)
    return 1;

  if (Memcpy(DHost, D, sizeof(DHost), MemcpyDeviceToHost) != Success)
    return 1;
  if (Memcpy(FHost, F, sizeof(FHost), MemcpyDeviceToHost) != Success)
    return 1;

  printf("double math A: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n", DHost[0],
         DHost[1], DHost[2], DHost[3], DHost[4], DHost[5], DHost[6], DHost[7]);
  // CHECK: double math A: 3.0 0.0 1.0 0.0 5.0 1.0 2.0 8.0
  printf("double math B: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n", DHost[8],
         DHost[9], DHost[10], DHost[11], DHost[12], DHost[13], DHost[14],
         DHost[15]);
  // CHECK: double math B: 7.0 2.0 1.0 2.0 1.0 2.0 2.0 0.0
  printf("float math A: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n", FHost[0],
         FHost[1], FHost[2], FHost[3], FHost[4], FHost[5], FHost[6], FHost[7]);
  // CHECK: float math A: 4.0 0.0 1.0 0.0 5.0 1.0 2.0 8.0
  printf("float math B: %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n", FHost[8],
         FHost[9], FHost[10], FHost[11], FHost[12], FHost[13], FHost[14],
         FHost[15]);
  // CHECK: float math B: 7.0 2.0 1.0 2.0 1.0 2.0 2.0 0.0

  Free(D);
  Free(F);
}
