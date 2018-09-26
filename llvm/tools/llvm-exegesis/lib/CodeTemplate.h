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
/// A CodeTemplate is a set of InstructionBuilders that may not be fully
/// specified (i.e. some variables are not yet set). This allows the
/// BenchmarkRunner to instantiate it many times with specific values to study
/// their impact on instruction's performance.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_CODETEMPLATE_H
#define LLVM_TOOLS_LLVM_EXEGESIS_CODETEMPLATE_H

#include "MCInstrDescView.h"

namespace exegesis {

struct CodeTemplate {
  CodeTemplate() = default;

  CodeTemplate(CodeTemplate &&);            // default
  CodeTemplate &operator=(CodeTemplate &&); // default
  CodeTemplate(const CodeTemplate &) = delete;
  CodeTemplate &operator=(const CodeTemplate &) = delete;

  // Some information about how this template has been created.
  std::string Info;
  // The list of the instructions for this template.
  std::vector<InstructionBuilder> Instructions;
  // If the template uses the provided scratch memory, the register in which
  // the pointer to this memory is passed in to the function.
  unsigned ScratchSpacePointerInReg = 0;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_CODETEMPLATE_H
