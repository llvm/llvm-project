#include "mathtest/DeviceContext.hpp"
#include "mathtest/ExhaustiveGenerator.hpp"
#include "mathtest/GpuMathTest.hpp"
#include "mathtest/IndexedRange.hpp"
#include "mathtest/TestRunner.hpp"

#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <limits>
#include <math.h>
#include <memory>

namespace mathtest {

template <> struct FunctionConfig<logf> {
  static constexpr llvm::StringRef Name = "logf";
  static constexpr llvm::StringRef KernelName = "logfKernel";

  // Source: The Khronos Group, The OpenCL C Specification v3.0.19, Sec. 7.4,
  //         Table 65, Khronos Registry [July 10, 2025].
  static constexpr uint64_t UlpTolerance = 3;
};
} // namespace mathtest

int main() {
  using namespace mathtest;

  // TODO: Add command-line arguments parsing for test configuration.
  auto Context = std::make_shared<DeviceContext>(/*DeviceId=*/0);
  const llvm::StringRef Provider = "llvm-libm";
  const llvm::StringRef DeviceBinsDirectory = DEVICE_CODE_PATH;

  GpuMathTest<logf> LogfTest(Context, Provider, DeviceBinsDirectory);

  IndexedRange<float> Range(/*Begin=*/0.0f,
                            /*End=*/std::numeric_limits<float>::infinity(),
                            /*Inclusive=*/true);
  ExhaustiveGenerator<float> Generator(Range);

  const auto Passed = runTest(LogfTest, Generator);

  return Passed ? EXIT_SUCCESS : EXIT_FAILURE;
}