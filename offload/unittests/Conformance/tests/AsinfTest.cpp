//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the conformance test of the asinf function.
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

namespace mathtest {

template <> struct FunctionConfig<asinf> {
  static constexpr llvm::StringRef Name = "asinf";
  static constexpr llvm::StringRef KernelName = "asinfKernel";

  // Source: The Khronos Group, The OpenCL C Specification v3.0.19, Sec. 7.4,
  //         Table 65, Khronos Registry [July 10, 2025].
  static constexpr uint64_t UlpTolerance = 4;
};
} // namespace mathtest

int main(int argc, const char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Conformance test of the asinf function");

  using namespace mathtest;

  IndexedRange<float> Range(/*Begin=*/-1.0f,
                            /*End=*/1.0f,
                            /*Inclusive=*/true);
  ExhaustiveGenerator<float> Generator(Range);

  const auto Configs = cl::getTestConfigs();
  const llvm::StringRef DeviceBinaryDir = DEVICE_BINARY_DIR;
  const bool IsVerbose = cl::IsVerbose;

  bool Passed = runTests<asinf>(Generator, Configs, DeviceBinaryDir, IsVerbose);

  return Passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
