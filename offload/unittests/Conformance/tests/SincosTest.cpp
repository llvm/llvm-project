//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the conformance test of the sincos function.
///
//===----------------------------------------------------------------------===//

#include "mathtest/CommandLineExtras.hpp"
#include "mathtest/IndexedRange.hpp"
#include "mathtest/RandomGenerator.hpp"
#include "mathtest/RandomState.hpp"
#include "mathtest/TestConfig.hpp"
#include "mathtest/TestRunner.hpp"

#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <math.h>

static inline double sincosSin(double X) {
  double SinX, CosX;
  sincos(X, &SinX, &CosX);
  return SinX;
}

static inline double sincosCos(double X) {
  double SinX, CosX;
  sincos(X, &SinX, &CosX);
  return CosX;
}

namespace mathtest {

template <> struct FunctionConfig<sincosSin> {
  static constexpr llvm::StringRef Name = "sincos (sin part)";
  static constexpr llvm::StringRef KernelName = "sincosSinKernel";

  // Source: The Khronos Group, The OpenCL C Specification v3.0.19, Sec. 7.4,
  //         Table 68, Khronos Registry [July 10, 2025].
  static constexpr uint64_t UlpTolerance = 4;
};

template <> struct FunctionConfig<sincosCos> {
  static constexpr llvm::StringRef Name = "sincos (cos part)";
  static constexpr llvm::StringRef KernelName = "sincosCosKernel";

  // Source: The Khronos Group, The OpenCL C Specification v3.0.19, Sec. 7.4,
  //         Table 68, Khronos Registry [July 10, 2025].
  static constexpr uint64_t UlpTolerance = 4;
};
} // namespace mathtest

int main(int argc, const char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Conformance test of the sincos function");

  using namespace mathtest;

  uint64_t Seed = 42;
  uint64_t Size = 1ULL << 32;
  IndexedRange<double> Range;
  RandomGenerator<double> Generator(SeedTy{Seed}, Size, Range);

  const auto Configs = cl::getTestConfigs();
  const llvm::StringRef DeviceBinaryDir = DEVICE_BINARY_DIR;
  const bool IsVerbose = cl::IsVerbose;

  bool SinPartPassed =
      runTests<sincosSin>(Generator, Configs, DeviceBinaryDir, IsVerbose);
  bool CosPartPassed =
      runTests<sincosCos>(Generator, Configs, DeviceBinaryDir, IsVerbose);

  return (SinPartPassed && CosPartPassed) ? EXIT_SUCCESS : EXIT_FAILURE;
}
