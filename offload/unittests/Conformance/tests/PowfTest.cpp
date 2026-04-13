//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the conformance test of the powf function.
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

static inline float powfRoundedExponent(float Base, float Exponent) {
  return powf(Base, roundf(Exponent));
}

namespace mathtest {

template <> struct FunctionConfig<powf> {
  static constexpr llvm::StringRef Name = "powf (real exponents)";
  static constexpr llvm::StringRef KernelName = "powfKernel";

  // Source: The Khronos Group, The OpenCL C Specification v3.0.19, Sec. 7.4,
  //         Table 65, Khronos Registry [July 10, 2025].
  static constexpr uint64_t UlpTolerance = 16;
};

template <> struct FunctionConfig<powfRoundedExponent> {
  static constexpr llvm::StringRef Name = "powf (integer exponents)";
  static constexpr llvm::StringRef KernelName = "powfRoundedExponentKernel";

  // Source: The Khronos Group, The OpenCL C Specification v3.0.19, Sec. 7.4,
  //         Table 65, Khronos Registry [July 10, 2025].
  static constexpr uint64_t UlpTolerance = 16;
};
} // namespace mathtest

int main(int argc, const char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Conformance test of the powf function");

  using namespace mathtest;

  uint64_t Size = 1ULL << 32;
  IndexedRange<float> RangeX;
  IndexedRange<float> RangeY;
  RandomGenerator<float, float> Generator0(SeedTy{42}, Size, RangeX, RangeY);
  RandomGenerator<float, float> Generator1(SeedTy{51}, Size, RangeX, RangeY);

  const auto Configs = cl::getTestConfigs();
  const llvm::StringRef DeviceBinaryDir = DEVICE_BINARY_DIR;
  const bool IsVerbose = cl::IsVerbose;

  bool RealExponentsPassed =
      runTests<powf>(Generator0, Configs, DeviceBinaryDir, IsVerbose);
  bool IntegerExponentsPassed = runTests<powfRoundedExponent>(
      Generator1, Configs, DeviceBinaryDir, IsVerbose);

  return (RealExponentsPassed && IntegerExponentsPassed) ? EXIT_SUCCESS
                                                         : EXIT_FAILURE;
}
