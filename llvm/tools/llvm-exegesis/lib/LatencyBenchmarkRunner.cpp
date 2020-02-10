//===-- LatencyBenchmarkRunner.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LatencyBenchmarkRunner.h"

#include "Target.h"
#include "BenchmarkRunner.h"

namespace llvm {
namespace exegesis {

LatencyBenchmarkRunner::LatencyBenchmarkRunner(const LLVMState &State,
                                               InstructionBenchmark::ModeE Mode)
    : BenchmarkRunner(State, Mode) {
  assert((Mode == InstructionBenchmark::Latency ||
          Mode == InstructionBenchmark::InverseThroughput) &&
         "invalid mode");
}

LatencyBenchmarkRunner::~LatencyBenchmarkRunner() = default;

Expected<std::vector<BenchmarkMeasure>> LatencyBenchmarkRunner::runMeasurements(
    const FunctionExecutor &Executor) const {
  // Cycle measurements include some overhead from the kernel. Repeat the
  // measure several times and take the minimum value.
  constexpr const int NumMeasurements = 30;
  int64_t MinValue = std::numeric_limits<int64_t>::max();
  const char *CounterName = State.getPfmCounters().CycleCounter;
  for (size_t I = 0; I < NumMeasurements; ++I) {
    auto ExpectedCounterValue = Executor.runAndMeasure(CounterName);
    if (!ExpectedCounterValue)
      return ExpectedCounterValue.takeError();
    if (*ExpectedCounterValue < MinValue)
      MinValue = *ExpectedCounterValue;
  }
  std::vector<BenchmarkMeasure> Result;
  switch (Mode) {
  case InstructionBenchmark::Latency:
    Result = {BenchmarkMeasure::Create("latency", MinValue)};
    break;
  case InstructionBenchmark::InverseThroughput:
    Result = {BenchmarkMeasure::Create("inverse_throughput", MinValue)};
    break;
  default:
    break;
  }
  return Result;
}

} // namespace exegesis
} // namespace llvm
