//===- llvm/MC/MCCodeEmitter.h - Instruction Encoding -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCCODEEMITTER_H
#define LLVM_MC_MCCODEEMITTER_H

namespace llvm {

class MCFixup;
class MCInst;
class MCSubtargetInfo;
class raw_ostream;
template<typename T> class SmallVectorImpl;

/// MCCodeEmitter - Generic instruction encoding interface.
class MCCodeEmitter {
protected: // Can only create subclasses.
  MCCodeEmitter();

  /// EncodeInstruction - Encode the given \p Inst to bytes on the output stream
  /// \p OS. Allows for an implementation of encodeInstruction that uses streams
  /// instead of a SmallVector.
  virtual void encodeInstruction(const MCInst &Inst, raw_ostream &OS,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const {}

public:
  MCCodeEmitter(const MCCodeEmitter &) = delete;
  MCCodeEmitter &operator=(const MCCodeEmitter &) = delete;
  virtual ~MCCodeEmitter();

  /// Lifetime management
  virtual void reset() {}

  /// Emit the prefixes of given instruction on the output stream.
  ///
  /// \param Inst a single low-level machine instruction.
  /// \param OS output stream.
  virtual void emitPrefix(const MCInst &Inst, raw_ostream &OS,
                          const MCSubtargetInfo &STI) const {}
  /// EncodeInstruction - Encode the given \p Inst to bytes and append to \p CB.
  virtual void encodeInstruction(const MCInst &Inst, SmallVectorImpl<char> &CB,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;
};

} // end namespace llvm

#endif // LLVM_MC_MCCODEEMITTER_H
