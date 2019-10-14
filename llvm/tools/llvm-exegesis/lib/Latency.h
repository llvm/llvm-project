//===-- Latency.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "Error.h"
#include "MCInstrDescView.h"
#include "SnippetGenerator.h"

namespace llvm {
namespace exegesis {

class LatencySnippetGenerator : public SnippetGenerator {
public:
  using SnippetGenerator::SnippetGenerator;
  ~LatencySnippetGenerator() override;

  Expected<std::vector<CodeTemplate>>
  generateCodeTemplates(const Instruction &Instr,
                        const BitVector &ForbiddenRegisters) const override;
};

class LatencyBenchmarkRunner : public BenchmarkRunner {
public:
  LatencyBenchmarkRunner(const LLVMState &State,
                         InstructionBenchmark::ModeE Mode);
  ~LatencyBenchmarkRunner() override;

private:
  Expected<std::vector<BenchmarkMeasure>>
  runMeasurements(const FunctionExecutor &Executor) const override;
};
} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_LATENCY_H
