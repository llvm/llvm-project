//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the conformance test of the log2f16 function.
///
//===----------------------------------------------------------------------===//

#include "mathtest/CommandLineExtras.hpp"
#include "mathtest/ExhaustiveGenerator.hpp"
#include "mathtest/IndexedRange.hpp"
#include "mathtest/Numerics.hpp"
#include "mathtest/TestConfig.hpp"
#include "mathtest/TestRunner.hpp"
#include "mathtest/TypeExtras.hpp"

#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <math.h>

using namespace mathtest;

extern "C" float16 log2f16(float16);

namespace mathtest {

template <> struct FunctionConfig<log2f16> {
  static constexpr llvm::StringRef Name = "log2f16";
  static constexpr llvm::StringRef KernelName = "log2f16Kernel";

  // Source: The Khronos Group, The OpenCL C Specification v3.0.19, Sec. 7.4,
  //         Table 69 (Full Profile), Khronos Registry [July 10, 2025].
  static constexpr uint64_t UlpTolerance = 2;
};
} // namespace mathtest

int main(int argc, const char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Conformance test of the log2f16 function");

  using namespace mathtest;

  IndexedRange<float16> Range(/*Begin=*/float16(0.0),
                              /*End=*/getMaxOrInf<float16>(),
                              /*Inclusive=*/true);
  ExhaustiveGenerator<float16> Generator(Range);

  const auto Configs = cl::getTestConfigs();
  const llvm::StringRef DeviceBinaryDir = DEVICE_BINARY_DIR;
  const bool IsVerbose = cl::IsVerbose;

  bool Passed =
      runTests<log2f16>(Generator, Configs, DeviceBinaryDir, IsVerbose);

  return Passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
