//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the conformance test of the sincosf function.
///
//===----------------------------------------------------------------------===//

#include "mathtest/CommandLineExtras.hpp"
#include "mathtest/ExhaustiveGenerator.hpp"
#include "mathtest/IndexedRange.hpp"
#include "mathtest/TestConfig.hpp"
#include "mathtest/TestRunner.hpp"

#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <math.h>

static inline float sincosfSin(float X) {
  float SinX, CosX;
  sincosf(X, &SinX, &CosX);
  return SinX;
}

static inline float sincosfCos(float X) {
  float SinX, CosX;
  sincosf(X, &SinX, &CosX);
  return CosX;
}

namespace mathtest {

template <> struct FunctionConfig<sincosfSin> {
  static constexpr llvm::StringRef Name = "sincosf (sin part)";
  static constexpr llvm::StringRef KernelName = "sincosfSinKernel";

  // Source: The Khronos Group, The OpenCL C Specification v3.0.19, Sec. 7.4,
  //         Table 65, Khronos Registry [July 10, 2025].
  static constexpr uint64_t UlpTolerance = 4;
};

template <> struct FunctionConfig<sincosfCos> {
  static constexpr llvm::StringRef Name = "sincosf (cos part)";
  static constexpr llvm::StringRef KernelName = "sincosfCosKernel";

  // Source: The Khronos Group, The OpenCL C Specification v3.0.19, Sec. 7.4,
  //         Table 65, Khronos Registry [July 10, 2025].
  static constexpr uint64_t UlpTolerance = 4;
};
} // namespace mathtest

int main(int argc, const char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Conformance test of the sincosf function");

  using namespace mathtest;

  IndexedRange<float> Range;
  ExhaustiveGenerator<float> Generator(Range);

  const auto Configs = cl::getTestConfigs();
  const llvm::StringRef DeviceBinaryDir = DEVICE_BINARY_DIR;
  const bool IsVerbose = cl::IsVerbose;

  bool SinPartPassed =
      runTests<sincosfSin>(Generator, Configs, DeviceBinaryDir, IsVerbose);
  bool CosPartPassed =
      runTests<sincosfCos>(Generator, Configs, DeviceBinaryDir, IsVerbose);

  return (SinPartPassed && CosPartPassed) ? EXIT_SUCCESS : EXIT_FAILURE;
}
