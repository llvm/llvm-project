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
  using BenchmarkRunner::BenchmarkRunner;
  ~LatencyBenchmarkRunner() override;

  llvm::Expected<BenchmarkConfiguration>
  generateConfiguration(unsigned Opcode) const;

private:
  llvm::Error isInfeasible(const llvm::MCInstrDesc &MCInstrDesc) const;

  llvm::Expected<BenchmarkConfiguration> generateSelfAliasingConfiguration(
      const Instruction &Instr,
      const AliasingConfigurations &SelfAliasing) const;

  llvm::Expected<BenchmarkConfiguration> generateTwoInstructionConfiguration(
      const Instruction &Instr,
      const AliasingConfigurations &SelfAliasing) const;

  InstructionBenchmark::ModeE getMode() const override;

  llvm::Expected<std::vector<BenchmarkConfiguration>>
  createConfigurations(unsigned OpcodeIndex) const override;

  std::vector<BenchmarkMeasure>
  runMeasurements(const ExecutableFunction &EF,
                  const unsigned NumRepetitions) const override;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_LATENCY_H
