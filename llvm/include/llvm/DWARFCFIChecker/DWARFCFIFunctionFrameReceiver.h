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

#ifndef LLVM_DWARFCFICHECKER_DWARFCFIFUNCTIONFRAMERECEIVER_H
#define LLVM_DWARFCFICHECKER_DWARFCFIFUNCTIONFRAMERECEIVER_H

#include "llvm/ADT/ArrayRef.h"

namespace llvm {

class MCCFIInstruction;
class MCContext;
class MCInst;

/// This abstract base class is an interface for receiving DWARF function frames
/// Call Frame Information. `DWARFCFIFunctionFrameStreamer` channels the
/// function frames information gathered from an `MCStreamer` using a pointer to
/// an instance of this class for the whole program.
class CFIFunctionFrameReceiver {
public:
  CFIFunctionFrameReceiver(const CFIFunctionFrameReceiver &) = delete;
  CFIFunctionFrameReceiver &
  operator=(const CFIFunctionFrameReceiver &) = delete;
  virtual ~CFIFunctionFrameReceiver() = default;

  CFIFunctionFrameReceiver(MCContext &Context) : Context(Context) {}

  MCContext &getContext() const { return Context; }

  virtual void startFunctionFrame(bool IsEH,
                                  ArrayRef<MCCFIInstruction> Prologue) {}
  /// Instructions are processed in the program order.
  virtual void
  emitInstructionAndDirectives(const MCInst &Inst,
                               ArrayRef<MCCFIInstruction> Directives) {}
  virtual void finishFunctionFrame() {}

private:
  MCContext &Context;
};

} // namespace llvm

#endif
