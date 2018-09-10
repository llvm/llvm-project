//===--- opencl_test.cpp - Tests for OpenCL and the Acxxel API ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "acxxel.h"
#include "gtest/gtest.h"

#include <array>
#include <cstring>

namespace {

static const char *SaxpyKernelSource = R"(
__kernel void saxpyKernel(float A, __global float *X, __global float *Y, int N) {
  int I = get_global_id(0);
  if (I < N)
    X[I] = A * X[I] + Y[I];
}
)";

TEST(OpenCL, Saxpy) {
  constexpr size_t Length = 3;

  float A = 2.f;
  std::array<float, Length> X = {{0.f, 1.f, 2.f}};
  std::array<float, Length> Y = {{3.f, 4.f, 5.f}};
  std::array<float, Length> Expected = {{3.f, 6.f, 9.f}};

  acxxel::Platform *OpenCL = acxxel::getOpenCLPlatform().getValue();
  acxxel::Stream Stream = OpenCL->createStream().takeValue();
  auto DeviceX = OpenCL->mallocD<float>(Length).takeValue();
  auto DeviceY = OpenCL->mallocD<float>(Length).takeValue();
  Stream.syncCopyHToD(X, DeviceX);
  Stream.syncCopyHToD(Y, DeviceY);
  acxxel::Program Program =
      OpenCL
          ->createProgramFromSource(acxxel::Span<const char>(
              SaxpyKernelSource, std::strlen(SaxpyKernelSource)))
          .takeValue();
  acxxel::Kernel Kernel = Program.createKernel("saxpyKernel").takeValue();
  float *RawX = static_cast<float *>(DeviceX);
  float *RawY = static_cast<float *>(DeviceY);
  int IntLength = Length;
  void *Arguments[] = {&A, &RawX, &RawY, &IntLength};
  size_t ArgumentSizes[] = {sizeof(float), sizeof(float *), sizeof(float *),
                            sizeof(int)};
  EXPECT_FALSE(
      Stream.asyncKernelLaunch(Kernel, Length, Arguments, ArgumentSizes)
          .takeStatus()
          .isError());
  Stream.syncCopyDToH(DeviceX, X);
  EXPECT_FALSE(Stream.sync().isError());

  EXPECT_EQ(X, Expected);
}

} // namespace
