//===-- X86.h ---------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// X86 target-specific setup.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_X86_H
#define LLVM_TOOLS_LLVM_EXEGESIS_X86_H

#include "BenchmarkRunner.h"
#include "LlvmState.h"

namespace exegesis {

class X86Filter : public BenchmarkRunner::InstructionFilter {
public:
  ~X86Filter() override;

  llvm::Error shouldRun(const LLVMState &State, unsigned Opcode) const override;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_X86_H
