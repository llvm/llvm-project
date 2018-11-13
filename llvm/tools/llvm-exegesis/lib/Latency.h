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
#include "MCInstrDescView.h"
#include "SnippetGenerator.h"

namespace llvm {
namespace exegesis {

class LatencySnippetGenerator : public SnippetGenerator {
public:
  LatencySnippetGenerator(const LLVMState &State) : SnippetGenerator(State) {}
  ~LatencySnippetGenerator() override;

  llvm::Expected<std::vector<CodeTemplate>>
  generateCodeTemplates(const Instruction &Instr) const override;
};

class LatencyBenchmarkRunner : public BenchmarkRunner {
public:
  LatencyBenchmarkRunner(const LLVMState &State)
      : BenchmarkRunner(State, InstructionBenchmark::Latency) {}
  ~LatencyBenchmarkRunner() override;

private:
  llvm::Expected<std::vector<BenchmarkMeasure>>
  runMeasurements(const FunctionExecutor &Executor) const override;
};
} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_LATENCY_H
