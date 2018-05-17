//===-- Latency.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A BenchmarkRunner implementation to measure instruction latencies.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_LATENCY_H
#define LLVM_TOOLS_LLVM_EXEGESIS_LATENCY_H

#include "BenchmarkRunner.h"

namespace exegesis {

class LatencyBenchmarkRunner : public BenchmarkRunner {
public:
  using BenchmarkRunner::BenchmarkRunner;
  ~LatencyBenchmarkRunner() override;

private:
  const char *getDisplayName() const override;

  llvm::Expected<std::vector<llvm::MCInst>>
  createSnippet(RegisterAliasingTrackerCache &RATC, unsigned OpcodeIndex,
                llvm::raw_ostream &Info) const override;

  std::vector<BenchmarkMeasure>
  runMeasurements(const ExecutableFunction &EF,
                  const unsigned NumRepetitions) const override;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_LATENCY_H
