//===-- SnippetGenerator.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the abstract SnippetGenerator class for generating code that allows
/// measuring a certain property of instructions (e.g. latency).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_SNIPPETGENERATOR_H
#define LLVM_TOOLS_LLVM_EXEGESIS_SNIPPETGENERATOR_H

#include "Assembler.h"
#include "BenchmarkCode.h"
#include "CodeTemplate.h"
#include "LlvmState.h"
#include "MCInstrDescView.h"
#include "RegisterAliasing.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Error.h"
#include <cstdlib>
#include <memory>
#include <vector>

namespace llvm {
namespace exegesis {

std::vector<CodeTemplate> getSingleton(CodeTemplate &&CT);

// Generates code templates that has a self-dependency.
llvm::Expected<std::vector<CodeTemplate>>
generateSelfAliasingCodeTemplates(const Instruction &Instr);

// Generates code templates without assignment constraints.
llvm::Expected<std::vector<CodeTemplate>>
generateUnconstrainedCodeTemplates(const Instruction &Instr,
                                   llvm::StringRef Msg);

// A class representing failures that happened during Benchmark, they are used
// to report informations to the user.
class SnippetGeneratorFailure : public llvm::StringError {
public:
  SnippetGeneratorFailure(const llvm::Twine &S);
};

// Common code for all benchmark modes.
class SnippetGenerator {
public:
  explicit SnippetGenerator(const LLVMState &State);

  virtual ~SnippetGenerator();

  // Calls generateCodeTemplate and expands it into one or more BenchmarkCode.
  llvm::Expected<std::vector<BenchmarkCode>>
  generateConfigurations(const Instruction &Instr) const;

  // Given a snippet, computes which registers the setup code needs to define.
  std::vector<RegisterValue> computeRegisterInitialValues(
      const std::vector<InstructionTemplate> &Snippet) const;

protected:
  const LLVMState &State;

private:
  // API to be implemented by subclasses.
  virtual llvm::Expected<std::vector<CodeTemplate>>
  generateCodeTemplates(const Instruction &Instr) const = 0;
};

// A global Random Number Generator to randomize configurations.
// FIXME: Move random number generation into an object and make it seedable for
// unit tests.
std::mt19937 &randomGenerator();

// Picks a random bit among the bits set in Vector and returns its index.
// Precondition: Vector must have at least one bit set.
size_t randomBit(const llvm::BitVector &Vector);

// Picks a random configuration, then selects a random def and a random use from
// it and finally set the selected values in the provided InstructionInstances.
void setRandomAliasing(const AliasingConfigurations &AliasingConfigurations,
                       InstructionTemplate &DefIB, InstructionTemplate &UseIB);

// Assigns a Random Value to all Variables in IT that are still Invalid.
// Do not use any of the registers in `ForbiddenRegs`.
void randomizeUnsetVariables(const llvm::BitVector &ForbiddenRegs,
                             InstructionTemplate &IT);

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_SNIPPETGENERATOR_H
