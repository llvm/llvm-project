//===-- Uops.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A BenchmarkRunner implementation to measure uop decomposition.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_UOPS_H
#define LLVM_TOOLS_LLVM_EXEGESIS_UOPS_H

#include "BenchmarkRunner.h"

namespace exegesis {

class UopsBenchmarkRunner : public BenchmarkRunner {
public:
  using BenchmarkRunner::BenchmarkRunner;
  ~UopsBenchmarkRunner() override;

  llvm::Expected<BenchmarkConfiguration>
  generateConfiguration(unsigned Opcode) const;

private:
  llvm::Error isInfeasible(const llvm::MCInstrDesc &MCInstrDesc) const;

  InstructionBenchmark::ModeE getMode() const override;

  llvm::Expected<std::vector<BenchmarkConfiguration>>
  createConfigurations(unsigned Opcode) const override;

  std::vector<BenchmarkMeasure>
  runMeasurements(const ExecutableFunction &EF,
                  const unsigned NumRepetitions) const override;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_UOPS_H
