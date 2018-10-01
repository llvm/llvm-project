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

namespace exegesis {

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
  generateConfigurations(unsigned Opcode) const;

  // Given a snippet, computes which registers the setup code needs to define.
  std::vector<RegisterValue> computeRegisterInitialValues(
      const std::vector<InstructionTemplate> &Snippet) const;

protected:
  const LLVMState &State;
  const RegisterAliasingTrackerCache RATC;

  // Generates a single code template that has a self-dependency.
  llvm::Expected<CodeTemplate>
  generateSelfAliasingCodeTemplate(const Instruction &Instr) const;
  // Generates a single code template without assignment constraints.
  llvm::Expected<CodeTemplate>
  generateUnconstrainedCodeTemplate(const Instruction &Instr,
                                    llvm::StringRef Msg) const;

private:
  // API to be implemented by subclasses.
  virtual llvm::Expected<CodeTemplate>
  generateCodeTemplate(unsigned Opcode) const = 0;
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

#endif // LLVM_TOOLS_LLVM_EXEGESIS_SNIPPETGENERATOR_H
