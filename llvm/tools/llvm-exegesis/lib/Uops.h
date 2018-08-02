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

  static constexpr const size_t kMinNumDifferentAddresses = 6;

private:
  llvm::Error isInfeasible(const llvm::MCInstrDesc &MCInstrDesc) const;

  std::vector<BenchmarkMeasure>
  runMeasurements(const ExecutableFunction &EF, ScratchSpace &Scratch,
                  const unsigned NumRepetitions) const override;

  // Instantiates memory operands within a snippet.
  // To make computations as parallel as possible, we generate independant
  // memory locations for instructions that load and store. If there are less
  // than kMinNumDifferentAddresses in the original snippet, we duplicate
  // instructions until there are this number of instructions.
  // For example, assuming kMinNumDifferentAddresses=5 and
  // getMaxMemoryAccessSize()=64, if the original snippet is:
  //   mov eax, [memory]
  // we might generate:
  //   mov eax, [rdi]
  //   mov eax, [rdi + 64]
  //   mov eax, [rdi + 128]
  //   mov eax, [rdi + 192]
  //   mov eax, [rdi + 256]
  // If the original snippet is:
  //   mov eax, [memory]
  //   add eax, [memory]
  // we might generate:
  //   mov eax, [rdi]
  //   add eax, [rdi + 64]
  //   mov eax, [rdi + 128]
  //   add eax, [rdi + 192]
  //   mov eax, [rdi + 256]
  void
  instantiateMemoryOperands(unsigned ScratchSpaceReg,
                            std::vector<InstructionInstance> &Snippet) const;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_UOPS_H
