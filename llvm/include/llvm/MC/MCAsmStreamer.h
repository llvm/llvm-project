//===- MCAsmStreamer.h - Base Class for Asm Streamers -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCAsmBaseStreamer class, a base class for streamers
// which emits assembly text.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMSTREAMER_H
#define LLVM_MC_MCASMSTREAMER_H

#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace llvm {

class MCAsmBackend;
class MCAsmInfo;
class MCAssembler;
class MCCodeEmitter;
class MCContext;
class MCInst;
class MCSubtargetInfo;

class MCAsmBaseStreamer : public MCStreamer {
protected:
  std::unique_ptr<MCAssembler> Assembler;
  SmallString<128> CommentToEmit;
  raw_svector_ostream CommentStream;
  raw_null_ostream NullStream;

  MCAsmBaseStreamer(MCContext &Context, std::unique_ptr<MCCodeEmitter> Emitter,
                std::unique_ptr<MCAsmBackend> AsmBackend);

public:
  /// Return a raw_ostream that comments can be written to.
  /// Unlike AddComment, you are required to terminate comments with \n if you
  /// use this method.
  raw_ostream &getCommentOS() override {
    if (!isVerboseAsm())
      return nulls(); // Discard comments unless in verbose asm mode.
    return CommentStream;
  }

  /// Add a comment showing the encoding of an instruction.
  /// \param Inst - The instruction to encode.
  /// \param STI - Subtarget information.
  void addEncodingComment(const MCInst &Inst, const MCSubtargetInfo &STI);

  MCAssembler &getAssembler() { return *Assembler; }
};

} // end namespace llvm

#endif // LLVM_MC_MCASMSTREAMER_H
