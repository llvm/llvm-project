//===- MCEncodingCommentHelper.h - Encoding Comment Helper ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helper functions for emitting encoding comments in
// assembly output.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCENCODINGCOMMENTHELPER_H
#define LLVM_MC_MCENCODINGCOMMENTHELPER_H

namespace llvm {

class MCInst;
class MCSubtargetInfo;
class MCAssembler;
class MCAsmInfo;
class raw_ostream;

namespace mc {

/// Emit an encoding comment for the given instruction.
///
/// This function encodes the instruction and emits a comment showing the
/// encoding bytes and any fixups that need to be applied.
///
/// \param OS - The output stream to write the comment to.
/// \param Inst - The instruction to encode.
/// \param STI - Subtarget information.
/// \param Assembler - The assembler containing the code emitter and backend.
/// \param MAI - Assembly info for formatting.
/// \param ForceLE - Force little-endian encoding regardless of target.
void emitEncodingComment(raw_ostream &OS, const MCInst &Inst,
                         const MCSubtargetInfo &STI, MCAssembler &Assembler,
                         const MCAsmInfo &MAI, bool ForceLE = false);
} // namespace mc
} // end namespace llvm

#endif // LLVM_MC_MCENCODINGCOMMENTHELPER_H
