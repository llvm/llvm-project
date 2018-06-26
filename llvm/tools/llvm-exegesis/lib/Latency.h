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

namespace exegesis {

class LatencyBenchmarkRunner : public BenchmarkRunner {
public:
  LatencyBenchmarkRunner(const LLVMState &State)
      : BenchmarkRunner(State, InstructionBenchmark::Latency) {}
  ~LatencyBenchmarkRunner() override;

  llvm::Expected<SnippetPrototype>
  generatePrototype(unsigned Opcode) const override;

private:
  llvm::Error isInfeasible(const llvm::MCInstrDesc &MCInstrDesc) const;

  llvm::Expected<SnippetPrototype> generateSelfAliasingPrototype(
      const Instruction &Instr,
      const AliasingConfigurations &SelfAliasing) const;

  llvm::Expected<SnippetPrototype> generateTwoInstructionPrototype(
      const Instruction &Instr,
      const AliasingConfigurations &SelfAliasing) const;

  std::vector<BenchmarkMeasure>
  runMeasurements(const ExecutableFunction &EF,
                  const unsigned NumRepetitions) const override;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_LATENCY_H
