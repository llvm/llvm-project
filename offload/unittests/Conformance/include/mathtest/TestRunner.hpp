//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the runTests function, which executes a
/// a suite of tests and print a formatted report for each.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_TESTRUNNER_HPP
#define MATHTEST_TESTRUNNER_HPP

#include "mathtest/DeviceContext.hpp"
#include "mathtest/GpuMathTest.hpp"
#include "mathtest/Numerics.hpp"
#include "mathtest/TestConfig.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <tuple>

namespace mathtest {
namespace detail {

template <auto Func>
void printPreamble(const TestConfig &Config, size_t Index,
                   size_t Total) noexcept {
  using FunctionConfig = FunctionConfig<Func>;

  llvm::errs() << "[" << (Index + 1) << "/" << Total << "] "
               << "Running conformance test '" << FunctionConfig::Name
               << "' with '" << Config.Provider << "' on '" << Config.Platform
               << "'\n";
  llvm::errs().flush();
}

template <typename T>
void printValue(llvm::raw_ostream &OS, const T &Value) noexcept {
  if constexpr (IsFloatingPoint_v<T>) {
    if constexpr (sizeof(T) < sizeof(float))
      OS << float(Value);
    else
      OS << Value;

    const FPBits<T> Bits(Value);
    OS << llvm::formatv(" (0x{0})", llvm::Twine::utohexstr(Bits.uintval()));
  } else {
    OS << Value;
  }
}

template <typename... Ts>
void printValues(llvm::raw_ostream &OS,
                 const std::tuple<Ts...> &ValuesTuple) noexcept {
  std::apply(
      [&OS](const auto &...Values) {
        bool IsFirst = true;
        auto PrintWithComma = [&](const auto &Value) {
          if (!IsFirst)
            OS << ", ";
          printValue(OS, Value);
          IsFirst = false;
        };
        (PrintWithComma(Values), ...);
      },
      ValuesTuple);
}

template <typename TestCaseType>
void printWorstFailingCase(llvm::raw_ostream &OS,
                           const TestCaseType &TestCase) noexcept {
  OS << "--- Worst Failing Case ---\n";
  OS << llvm::formatv("  {0,-14} : ", "Input(s)");
  printValues(OS, TestCase.Inputs);
  OS << "\n";

  OS << llvm::formatv("  {0,-14} : ", "Actual");
  printValue(OS, TestCase.Actual);
  OS << "\n";

  OS << llvm::formatv("  {0,-14} : ", "Expected");
  printValue(OS, TestCase.Expected);
  OS << "\n";
}

template <typename TestType, typename ResultType>
void printReport(const TestType &Test, const ResultType &Result,
                 const std::chrono::steady_clock::duration &Duration) noexcept {
  using FunctionConfig = typename TestType::FunctionConfig;

  const auto Context = Test.getContext();
  const auto ElapsedMilliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(Duration).count();
  const bool Passed = Result.hasPassed();

  llvm::errs() << llvm::formatv("=== Test Report for '{0}' === \n",
                                FunctionConfig::Name);
  llvm::errs() << llvm::formatv("{0,-17}: {1}\n", "Provider",
                                Test.getProvider());
  llvm::errs() << llvm::formatv("{0,-17}: {1}\n", "Platform",
                                Context->getPlatform());
  llvm::errs() << llvm::formatv("{0,-17}: {1}\n", "Device", Context->getName());
  llvm::errs() << llvm::formatv("{0,-17}: {1} ms\n", "Elapsed time",
                                ElapsedMilliseconds);
  llvm::errs() << llvm::formatv("{0,-17}: {1}\n", "ULP tolerance",
                                FunctionConfig::UlpTolerance);
  llvm::errs() << llvm::formatv("{0,-17}: {1}\n", "Max ULP distance",
                                Result.getMaxUlpDistance());
  llvm::errs() << llvm::formatv("{0,-17}: {1}\n", "Test cases",
                                Result.getTestCaseCount());
  llvm::errs() << llvm::formatv("{0,-17}: {1}\n", "Failures",
                                Result.getFailureCount());
  llvm::errs() << llvm::formatv("{0,-17}: {1}\n", "Status",
                                Passed ? "PASSED" : "FAILED");

  if (const auto &Worst = Result.getWorstFailingCase())
    printWorstFailingCase(llvm::errs(), Worst.value());

  llvm::errs().flush();
}

template <auto Func, typename TestType = GpuMathTest<Func>>
[[nodiscard]] llvm::Expected<bool>
runTest(typename TestType::GeneratorType &Generator, const TestConfig &Config,
        llvm::StringRef DeviceBinaryDir) {
  const auto &Platforms = getPlatforms();

  if (!llvm::any_of(Platforms, [&](llvm::StringRef Platform) {
        return Platform.equals_insensitive(Config.Platform);
      }))
    return llvm::createStringError("Platform '" + Config.Platform +
                                   "' is not available on this system");

  auto Context =
      std::make_shared<DeviceContext>(Config.Platform, /*DeviceId=*/0);
  auto ExpectedTest =
      TestType::create(Context, Config.Provider, DeviceBinaryDir);

  if (!ExpectedTest)
    return ExpectedTest.takeError();

  const auto StartTime = std::chrono::steady_clock::now();

  auto Result = ExpectedTest->run(Generator);

  const auto EndTime = std::chrono::steady_clock::now();
  const auto Duration = EndTime - StartTime;

  printReport(*ExpectedTest, Result, Duration);

  return Result.hasPassed();
}
} // namespace detail

template <auto Func, typename TestType = GpuMathTest<Func>>
[[nodiscard]] bool runTests(typename TestType::GeneratorType &Generator,
                            const llvm::SmallVector<TestConfig, 4> &Configs,
                            llvm::StringRef DeviceBinaryDir,
                            bool IsVerbose = false) {
  const size_t NumConfigs = Configs.size();

  if (NumConfigs == 0)
    llvm::errs() << "There is no test configuration to run a test\n";

  bool Passed = true;

  for (const auto &[Index, Config] : llvm::enumerate(Configs)) {
    detail::printPreamble<Func>(Config, Index, NumConfigs);

    Generator.reset();

    auto ExpectedPassed =
        detail::runTest<Func, TestType>(Generator, Config, DeviceBinaryDir);

    if (!ExpectedPassed) {
      const auto Details = llvm::toString(ExpectedPassed.takeError());
      llvm::errs()
          << "WARNING: Conformance test not supported on this system\n";

      if (IsVerbose)
        llvm::errs() << "Details: " << Details << "\n";
    } else {
      Passed &= *ExpectedPassed;
    }

    llvm::errs() << "\n";
  }

  return Passed;
}
} // namespace mathtest

#endif // MATHTEST_TESTRUNNER_HPP
