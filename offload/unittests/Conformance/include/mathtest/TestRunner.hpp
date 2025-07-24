//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the runTest function, which executes a
/// test instance and prints a formatted report of the results.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_TESTRUNNER_HPP
#define MATHTEST_TESTRUNNER_HPP

#include "mathtest/Numerics.hpp"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <tuple>

namespace mathtest {
namespace detail {

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
        auto Print = [&](const auto &Value) {
          if (!IsFirst)
            OS << ", ";
          printValue(OS, Value);
          IsFirst = false;
        };
        (Print(Values), ...);
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

  if (auto Worst = Result.getWorstFailingCase())
    printWorstFailingCase(llvm::errs(), Worst.value());

  llvm::errs().flush();
}
} // namespace detail

template <typename TestType>
[[nodiscard]] bool
runTest(const TestType &Test,
        typename TestType::GeneratorType &Generator) noexcept {
  const auto StartTime = std::chrono::steady_clock::now();

  auto Result = Test.run(Generator);

  const auto EndTime = std::chrono::steady_clock::now();
  const auto Duration = EndTime - StartTime;

  detail::printReport(Test, Result, Duration);

  return Result.hasPassed();
}
} // namespace mathtest

#endif // MATHTEST_TESTRUNNER_HPP
