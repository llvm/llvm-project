//===-- llvm/MC/MCCodeEmitter.h - Instruction Encoding ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCCODEEMITTER_H
#define LLVM_MC_MCCODEEMITTER_H

#include "llvm/Support/Compiler.h"

namespace llvm {
class MCFixup;
class MCInst;
class raw_ostream;
template<typename T> class SmallVectorImpl;

/// MCCodeEmitter - Generic instruction encoding interface.
class MCCodeEmitter {
private:
  MCCodeEmitter(const MCCodeEmitter &) LLVM_DELETED_FUNCTION;
  void operator=(const MCCodeEmitter &) LLVM_DELETED_FUNCTION;
protected: // Can only create subclasses.
  MCCodeEmitter();

public:
  virtual ~MCCodeEmitter();

  /// EncodeInstruction - Encode the given \p Inst to bytes on the output
  /// stream \p OS.
  virtual void EncodeInstruction(const MCInst &Inst, raw_ostream &OS,
                                 SmallVectorImpl<MCFixup> &Fixups) const = 0;
};

} // End llvm namespace

#endif
