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

  const llvm::StringRef Platform = PLATFORM;
  auto Context = std::make_shared<DeviceContext>(Platform, /*DeviceId=*/0);

  const llvm::StringRef Provider = PROVIDER;
  const llvm::StringRef DeviceBinaryDir = DEVICE_BINARY_DIR;
  GpuMathTest<logf> LogfTest(Context, Provider, DeviceBinaryDir);

  IndexedRange<float> Range(/*Begin=*/0.0f,
                            /*End=*/std::numeric_limits<float>::infinity(),
                            /*Inclusive=*/true);
  ExhaustiveGenerator<float> Generator(Range);

  const auto Passed = runTest(LogfTest, Generator);

  return Passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
