//===--- simple_example.cu - Simple example of using Acxxel ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// This file is a simple example of using Acxxel.
///
//===----------------------------------------------------------------------===//

/// [Example simple saxpy]
#include "acxxel.h"

#include <array>
#include <cstdio>
#include <cstdlib>

// A standard CUDA kernel.
__global__ void saxpyKernel(float A, float *X, float *Y, int N) {
  int I = (blockDim.x * blockIdx.x) + threadIdx.x;
  if (I < N)
    X[I] = A * X[I] + Y[I];
}

// A host library wrapping the CUDA kernel. All Acxxel calls are in here.
template <size_t N>
void saxpy(float A, std::array<float, N> &X, const std::array<float, N> &Y) {
  // Get the CUDA platform and make a CUDA stream.
  acxxel::Platform *CUDA = acxxel::getCUDAPlatform().getValue();
  acxxel::Stream Stream = CUDA->createStream().takeValue();

  // Allocate space for device arrays.
  auto DeviceX = CUDA->mallocD<float>(N).takeValue();
  auto DeviceY = CUDA->mallocD<float>(N).takeValue();

  // Copy X and Y out to the device.
  Stream.syncCopyHToD(X, DeviceX).syncCopyHToD(Y, DeviceY);

  // Launch the kernel using triple-chevron notation.
  saxpyKernel<<<1, N, 0, Stream>>>(A, DeviceX, DeviceY, N);

  // Copy the results back to the host.
  acxxel::Status Status = Stream.syncCopyDToH(DeviceX, X).takeStatus();

  // Check for any errors.
  if (Status.isError()) {
    std::fprintf(stderr, "Error performing acxxel saxpy: %s\n",
                 Status.getMessage().c_str());
    std::exit(EXIT_FAILURE);
  }
}
/// [Example simple saxpy]

/// [Example CUDA simple saxpy]
template <size_t N>
void cudaSaxpy(float A, std::array<float, N> &X, std::array<float, N> &Y) {
  // This size is needed all over the place, so give it a name.
  constexpr size_t Size = N * sizeof(float);

  // Allocate space for device arrays.
  float *DeviceX;
  float *DeviceY;
  cudaMalloc(&DeviceX, Size);
  cudaMalloc(&DeviceY, Size);

  // Copy X and Y out to the device.
  cudaMemcpy(DeviceX, X.data(), Size, cudaMemcpyHostToDevice);
  cudaMemcpy(DeviceY, Y.data(), Size, cudaMemcpyHostToDevice);

  // Launch the kernel using triple-chevron notation.
  saxpyKernel<<<1, N>>>(A, DeviceX, DeviceY, N);

  // Copy the results back to the host.
  cudaMemcpy(X.data(), DeviceX, Size, cudaMemcpyDeviceToHost);

  // Free resources.
  cudaFree(DeviceX);
  cudaFree(DeviceY);

  // Check for any errors.
  cudaError_t Error = cudaGetLastError();
  if (Error) {
    std::fprintf(stderr, "Error performing cudart saxpy: %s\n",
                 cudaGetErrorString(Error));
    std::exit(EXIT_FAILURE);
  }
}
/// [Example CUDA simple saxpy]

template <typename F> void testSaxpy(F &&SaxpyFunction) {
  float A = 2.f;
  std::array<float, 3> X = {{0.f, 1.f, 2.f}};
  std::array<float, 3> Y = {{3.f, 4.f, 5.f}};
  std::array<float, 3> Expected = {{3.f, 6.f, 9.f}};
  SaxpyFunction(A, X, Y);
  for (int I = 0; I < 3; ++I)
    if (X[I] != Expected[I]) {
      std::fprintf(stderr, "Result mismatch at index %d, %f != %f\n", I, X[I],
                   Expected[I]);
      std::exit(EXIT_FAILURE);
    }
}

int main() {
  testSaxpy(saxpy<3>);
  testSaxpy(cudaSaxpy<3>);
}
