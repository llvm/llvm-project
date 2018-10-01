//===-- CodeTemplate.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A set of structures and functions to craft instructions for the
/// SnippetGenerator.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_CODETEMPLATE_H
#define LLVM_TOOLS_LLVM_EXEGESIS_CODETEMPLATE_H

#include "MCInstrDescView.h"

namespace exegesis {

// A template for an Instruction holding values for each of its Variables.
struct InstructionTemplate {
  InstructionTemplate(const Instruction &Instr);

  InstructionTemplate(const InstructionTemplate &);            // default
  InstructionTemplate &operator=(const InstructionTemplate &); // default
  InstructionTemplate(InstructionTemplate &&);                 // default
  InstructionTemplate &operator=(InstructionTemplate &&);      // default

  unsigned getOpcode() const;
  llvm::MCOperand &getValueFor(const Variable &Var);
  const llvm::MCOperand &getValueFor(const Variable &Var) const;
  llvm::MCOperand &getValueFor(const Operand &Op);
  const llvm::MCOperand &getValueFor(const Operand &Op) const;
  bool hasImmediateVariables() const;

  // Builds an llvm::MCInst from this InstructionTemplate setting its operands
  // to the corresponding variable values. Precondition: All VariableValues must
  // be set.
  llvm::MCInst build() const;

  Instruction Instr;
  llvm::SmallVector<llvm::MCOperand, 4> VariableValues;
};

// A CodeTemplate is a set of InstructionTemplates that may not be fully
// specified (i.e. some variables are not yet set). This allows the
// BenchmarkRunner to instantiate it many times with specific values to study
// their impact on instruction's performance.
struct CodeTemplate {
  CodeTemplate() = default;

  CodeTemplate(CodeTemplate &&);            // default
  CodeTemplate &operator=(CodeTemplate &&); // default
  CodeTemplate(const CodeTemplate &) = delete;
  CodeTemplate &operator=(const CodeTemplate &) = delete;

  // Some information about how this template has been created.
  std::string Info;
  // The list of the instructions for this template.
  std::vector<InstructionTemplate> Instructions;
  // If the template uses the provided scratch memory, the register in which
  // the pointer to this memory is passed in to the function.
  unsigned ScratchSpacePointerInReg = 0;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_CODETEMPLATE_H
