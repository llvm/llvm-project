//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the conformance test of the hypotf16 function.
///
//===----------------------------------------------------------------------===//

#include "mathtest/CommandLineExtras.hpp"
#include "mathtest/ExhaustiveGenerator.hpp"
#include "mathtest/IndexedRange.hpp"
#include "mathtest/TestConfig.hpp"
#include "mathtest/TestRunner.hpp"
#include "mathtest/TypeExtras.hpp"

#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <math.h>

using namespace mathtest;

extern "C" float16 hypotf16(float16, float16);

namespace mathtest {

template <> struct FunctionConfig<hypotf16> {
  static constexpr llvm::StringRef Name = "hypotf16";
  static constexpr llvm::StringRef KernelName = "hypotf16Kernel";

  // Source: The Khronos Group, The OpenCL C Specification v3.0.19, Sec. 7.4,
  //         Table 69 (Full Profile), Khronos Registry [July 10, 2025].
  static constexpr uint64_t UlpTolerance = 2;
};
} // namespace mathtest

int main(int argc, const char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Conformance test of the hypotf16 function");

  IndexedRange<float16> RangeX;
  IndexedRange<float16> RangeY;
  ExhaustiveGenerator<float16, float16> Generator(RangeX, RangeY);

  const auto Configs = cl::getTestConfigs();
  const llvm::StringRef DeviceBinaryDir = DEVICE_BINARY_DIR;
  const bool IsVerbose = cl::IsVerbose;

  bool Passed =
      runTests<hypotf16>(Generator, Configs, DeviceBinaryDir, IsVerbose);

  return Passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
