//===- llvm/IR/FunctionInstructionPrinter.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_FUNCTIONINSTRUCTIONPRINTER_H
#define LLVM_IR_FUNCTIONINSTRUCTIONPRINTER_H

#include "llvm/Support/Compiler.h"
#include <memory>

namespace llvm {

class Function;
class Instruction;
class ModuleSlotTracker;
class raw_ostream;

/// Print multiple instructions from one function without repeatedly creating
/// the underlying assembly writer.
///
/// OS, MST, and F must remain valid for this object's lifetime. The function's
/// IR must not be modified, and MST must not incorporate another function while
/// the printer is in use. Each instruction is forwarded to OS before
/// printInstruction returns, so callers can safely interleave other output.
class LLVM_ABI FunctionInstructionPrinter {
  struct Impl;
  std::unique_ptr<Impl> P;

public:
  FunctionInstructionPrinter(raw_ostream &OS, ModuleSlotTracker &MST,
                             const Function &F, bool IsForDebug = false);
  ~FunctionInstructionPrinter();

  FunctionInstructionPrinter(const FunctionInstructionPrinter &) = delete;
  FunctionInstructionPrinter &
  operator=(const FunctionInstructionPrinter &) = delete;

  void printInstruction(const Instruction &I);
};

} // end namespace llvm

#endif // LLVM_IR_FUNCTIONINSTRUCTIONPRINTER_H
