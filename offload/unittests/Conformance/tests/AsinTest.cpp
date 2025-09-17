//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the conformance test of the asin function.
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

namespace {

// Disambiguate the overloaded 'asin' function to select the double version
constexpr auto asind // NOLINT(readability-identifier-naming)
    = static_cast<double (*)(double)>(asin);
} // namespace

namespace mathtest {

template <> struct FunctionConfig<asind> {
  static constexpr llvm::StringRef Name = "asin";
  static constexpr llvm::StringRef KernelName = "asinKernel";

  // Source: The Khronos Group, The OpenCL C Specification v3.0.19, Sec. 7.4,
  //         Table 68, Khronos Registry [July 10, 2025].
  static constexpr uint64_t UlpTolerance = 4;
};
} // namespace mathtest

int main(int argc, const char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Conformance test of the asin function");

  using namespace mathtest;

  uint64_t Seed = 42;
  uint64_t Size = 1ULL << 32;
  IndexedRange<double> Range(/*Begin=*/-1.0,
                             /*End=*/1.0,
                             /*Inclusive=*/true);
  RandomGenerator<double> Generator(SeedTy{Seed}, Size, Range);

  const auto Configs = cl::getTestConfigs();
  const llvm::StringRef DeviceBinaryDir = DEVICE_BINARY_DIR;
  const bool IsVerbose = cl::IsVerbose;

  bool Passed = runTests<asind>(Generator, Configs, DeviceBinaryDir, IsVerbose);

  return Passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
