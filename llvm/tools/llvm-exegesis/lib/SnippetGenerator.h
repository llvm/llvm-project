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
      const std::vector<InstructionBuilder> &Snippet) const;

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

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_SNIPPETGENERATOR_H
