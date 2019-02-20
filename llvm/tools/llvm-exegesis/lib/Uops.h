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
#include "SnippetGenerator.h"

namespace llvm {
namespace exegesis {

class UopsSnippetGenerator : public SnippetGenerator {
public:
  UopsSnippetGenerator(const LLVMState &State) : SnippetGenerator(State) {}
  ~UopsSnippetGenerator() override;

  llvm::Expected<std::vector<CodeTemplate>>
  generateCodeTemplates(const Instruction &Instr) const override;

  static constexpr const size_t kMinNumDifferentAddresses = 6;

private:
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
  void instantiateMemoryOperands(
      unsigned ScratchSpaceReg,
      std::vector<InstructionTemplate> &SnippetTemplate) const;
};

class UopsBenchmarkRunner : public BenchmarkRunner {
public:
  UopsBenchmarkRunner(const LLVMState &State)
      : BenchmarkRunner(State, InstructionBenchmark::Uops) {}
  ~UopsBenchmarkRunner() override;

  static constexpr const size_t kMinNumDifferentAddresses = 6;

private:
  llvm::Expected<std::vector<BenchmarkMeasure>>
  runMeasurements(const FunctionExecutor &Executor) const override;
};

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_UOPS_H
