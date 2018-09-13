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

namespace exegesis {

class LatencySnippetGenerator : public SnippetGenerator {
public:
  LatencySnippetGenerator(const LLVMState &State) : SnippetGenerator(State) {}
  ~LatencySnippetGenerator() override;

  llvm::Expected<CodeTemplate>
  generateCodeTemplate(unsigned Opcode) const override;

private:
  llvm::Error isInfeasible(const llvm::MCInstrDesc &MCInstrDesc) const;

  llvm::Expected<CodeTemplate>
  generateTwoInstructionPrototype(const Instruction &Instr) const;
};

class LatencyBenchmarkRunner : public BenchmarkRunner {
public:
  LatencyBenchmarkRunner(const LLVMState &State)
      : BenchmarkRunner(State, InstructionBenchmark::Latency) {}
  ~LatencyBenchmarkRunner() override;

private:
  std::vector<BenchmarkMeasure>
  runMeasurements(const ExecutableFunction &EF, ScratchSpace &Scratch,
                  const unsigned NumRepetitions) const override;

  virtual const char *getCounterName() const;
};
} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_LATENCY_H
