//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares FunctionUnitAnalyzer class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNWINDINFOCHECKER_FUNCTIONUNITANALYZER_H
#define LLVM_UNWINDINFOCHECKER_FUNCTIONUNITANALYZER_H

#include "llvm/ADT/ArrayRef.h"

namespace llvm {

class MCCFIInstruction;
class MCContext;
class MCInst;

class FunctionUnitAnalyzer {
private:
  MCContext &Context;

public:
  FunctionUnitAnalyzer(const FunctionUnitAnalyzer &) = delete;
  FunctionUnitAnalyzer &operator=(const FunctionUnitAnalyzer &) = delete;
  virtual ~FunctionUnitAnalyzer();

  FunctionUnitAnalyzer(MCContext &Context) : Context(Context) {}

  MCContext &getContext() const { return Context; }

  virtual void startFunctionUnit(bool IsEH,
                                 ArrayRef<MCCFIInstruction> Prologue);
  virtual void
  emitInstructionAndDirectives(const MCInst &Inst,
                               ArrayRef<MCCFIInstruction> Directives);
  virtual void finishFunctionUnit();
};

} // namespace llvm

#endif
