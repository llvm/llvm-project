//===- llvm/MC/MCCodeEmitter.h - Instruction Encoding -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCCODEEMITTER_H
#define LLVM_MC_MCCODEEMITTER_H

#include "llvm/Support/Compiler.h"

namespace llvm {

class MCFixup;
class MCInst;
class MCSubtargetInfo;
template<typename T> class SmallVectorImpl;

/// MCCodeEmitter - Generic instruction encoding interface.
class LLVM_ABI MCCodeEmitter {
protected: // Can only create subclasses.
  MCCodeEmitter();

public:
  MCCodeEmitter(const MCCodeEmitter &) = delete;
  MCCodeEmitter &operator=(const MCCodeEmitter &) = delete;
  virtual ~MCCodeEmitter();

  /// Lifetime management
  virtual void reset() {}

  /// Encode the given \p Inst to bytes and append to \p CB.
  virtual void encodeInstruction(const MCInst &Inst, SmallVectorImpl<char> &CB,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const = 0;

protected:
  // Helper function used by CodeEmitterGen for error reporting.
  [[noreturn]] static void reportUnsupportedInst(const MCInst &Inst);
  [[noreturn]] static void reportUnsupportedOperand(const MCInst &Inst,
                                                    unsigned OpNum);
};

} // end namespace llvm

#endif // LLVM_MC_MCCODEEMITTER_H
