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
  UopsBenchmarkRunner(const LLVMState &State)
      : BenchmarkRunner(State, InstructionBenchmark::Uops) {}
  ~UopsBenchmarkRunner() override;

  llvm::Expected<SnippetPrototype>
  generatePrototype(unsigned Opcode) const override;

private:
  llvm::Error isInfeasible(const llvm::MCInstrDesc &MCInstrDesc) const;

  std::vector<BenchmarkMeasure>
  runMeasurements(const ExecutableFunction &EF,
                  const unsigned NumRepetitions) const override;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_UOPS_H
