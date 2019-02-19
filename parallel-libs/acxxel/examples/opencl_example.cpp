//===--- opencl_example.cpp - Example of using Acxxel with OpenCL ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file is an example of using OpenCL with Acxxel.
///
//===----------------------------------------------------------------------===//

#include "acxxel.h"

#include <array>
#include <cstdio>
#include <cstring>

static const char *SaxpyKernelSource = R"(
__kernel void saxpyKernel(float A, __global float *X, __global float *Y, int N) {
  int I = get_global_id(0);
  if (I < N)
    X[I] = A * X[I] + Y[I];
}
)";

template <size_t N>
void saxpy(float A, std::array<float, N> &X, const std::array<float, N> &Y) {
  acxxel::Platform *OpenCL = acxxel::getOpenCLPlatform().getValue();
  acxxel::Stream Stream = OpenCL->createStream().takeValue();
  auto DeviceX = OpenCL->mallocD<float>(N).takeValue();
  auto DeviceY = OpenCL->mallocD<float>(N).takeValue();
  Stream.syncCopyHToD(X, DeviceX).syncCopyHToD(Y, DeviceY);
  acxxel::Program Program =
      OpenCL
          ->createProgramFromSource(acxxel::Span<const char>(
              SaxpyKernelSource, std::strlen(SaxpyKernelSource)))
          .takeValue();
  acxxel::Kernel Kernel = Program.createKernel("saxpyKernel").takeValue();
  float *RawX = static_cast<float *>(DeviceX);
  float *RawY = static_cast<float *>(DeviceY);
  int IntLength = N;
  void *Arguments[] = {&A, &RawX, &RawY, &IntLength};
  size_t ArgumentSizes[] = {sizeof(float), sizeof(float *), sizeof(float *),
                            sizeof(int)};
  acxxel::Status Status =
      Stream.asyncKernelLaunch(Kernel, N, Arguments, ArgumentSizes)
          .syncCopyDToH(DeviceX, X)
          .sync();
  if (Status.isError()) {
    std::fprintf(stderr, "Error during saxpy: %s\n",
                 Status.getMessage().c_str());
    std::exit(EXIT_FAILURE);
  }
}

int main() {
  float A = 2.f;
  std::array<float, 3> X{{0.f, 1.f, 2.f}};
  std::array<float, 3> Y{{3.f, 4.f, 5.f}};
  std::array<float, 3> Expected{{3.f, 6.f, 9.f}};
  saxpy(A, X, Y);
  for (int I = 0; I < 3; ++I)
    if (X[I] != Expected[I]) {
      std::fprintf(stderr, "Mismatch at position %d, %f != %f\n", I, X[I],
                   Expected[I]);
      std::exit(EXIT_FAILURE);
    }
}
