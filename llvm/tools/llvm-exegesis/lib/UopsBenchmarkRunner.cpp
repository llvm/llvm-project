//===-- UopsBenchmarkRunner.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UopsBenchmarkRunner.h"

#include "Target.h"

namespace llvm {
namespace exegesis {

UopsBenchmarkRunner::~UopsBenchmarkRunner() = default;

Expected<std::vector<BenchmarkMeasure>>
UopsBenchmarkRunner::runMeasurements(const FunctionExecutor &Executor) const {
  std::vector<BenchmarkMeasure> Result;
  const PfmCountersInfo &PCI = State.getPfmCounters();
  // Uops per port.
  for (const auto *IssueCounter = PCI.IssueCounters,
                  *IssueCounterEnd = PCI.IssueCounters + PCI.NumIssueCounters;
       IssueCounter != IssueCounterEnd; ++IssueCounter) {
    if (!IssueCounter->Counter)
      continue;
    auto ExpectedCounterValue = Executor.runAndSample(IssueCounter->Counter);
    if (!ExpectedCounterValue)
      return ExpectedCounterValue.takeError();
    Result.push_back(BenchmarkMeasure::Create(IssueCounter->ProcResName,
                                              (*ExpectedCounterValue)[0]));
  }
  // NumMicroOps.
  if (const char *const UopsCounter = PCI.UopsCounter) {
    auto ExpectedCounterValue = Executor.runAndSample(UopsCounter);
    if (!ExpectedCounterValue)
      return ExpectedCounterValue.takeError();
    Result.push_back(
        BenchmarkMeasure::Create("NumMicroOps", (*ExpectedCounterValue)[0]));
  }
  return std::move(Result);
}

} // namespace exegesis
} // namespace llvm
