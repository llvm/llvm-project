//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares CFIFunctionFrameReceiver class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNWINDINFOCHECKER_DWARFCFIFUNCTIONFRAMERECEIVER_H
#define LLVM_UNWINDINFOCHECKER_DWARFCFIFUNCTIONFRAMERECEIVER_H

#include "llvm/ADT/ArrayRef.h"

namespace llvm {

class MCCFIInstruction;
class MCContext;
class MCInst;

class CFIFunctionFrameReceiver {
private:
  MCContext &Context;

public:
  CFIFunctionFrameReceiver(const CFIFunctionFrameReceiver &) = delete;
  CFIFunctionFrameReceiver &
  operator=(const CFIFunctionFrameReceiver &) = delete;
  virtual ~CFIFunctionFrameReceiver();

  CFIFunctionFrameReceiver(MCContext &Context) : Context(Context) {}

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
