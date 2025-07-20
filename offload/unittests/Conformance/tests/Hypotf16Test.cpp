#include "mathtest/TypeExtras.hpp"

#ifdef MATHTEST_HAS_FLOAT16
#include "mathtest/DeviceContext.hpp"
#include "mathtest/ExhaustiveGenerator.hpp"
#include "mathtest/GpuMathTest.hpp"
#include "mathtest/IndexedRange.hpp"
#include "mathtest/TestRunner.hpp"

#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <math.h>
#include <memory>

using namespace mathtest;

extern "C" {

float16 hypotf16(float16, float16);
}

namespace mathtest {

template <> struct FunctionConfig<hypotf16> {
  static constexpr llvm::StringRef Name = "hypotf16";
  static constexpr llvm::StringRef KernelName = "hypotf16Kernel";

  // Source: The Khronos Group, The OpenCL C Specification v3.0.19, Sec. 7.4,
  //         Table 69 (Full Profile), Khronos Registry [July 10, 2025].
  static constexpr uint64_t UlpTolerance = 2;
};
} // namespace mathtest

int main() {
  const llvm::StringRef Platform = PLATFORM;
  auto Context = std::make_shared<DeviceContext>(Platform, /*DeviceId=*/0);

  const llvm::StringRef Provider = PROVIDER;
  const llvm::StringRef DeviceBinaryDir = DEVICE_BINARY_DIR;
  GpuMathTest<hypotf16> Hypotf16Test(Context, Provider, DeviceBinaryDir);

  IndexedRange<float16> RangeX;
  IndexedRange<float16> RangeY;
  ExhaustiveGenerator<float16, float16> Generator(RangeX, RangeY);

  const auto Passed = runTest(Hypotf16Test, Generator);

  return Passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif // MATHTEST_HAS_FLOAT16
