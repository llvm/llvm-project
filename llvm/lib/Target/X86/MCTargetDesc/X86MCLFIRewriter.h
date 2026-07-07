//===- X86MCLFIRewriter.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the X86MCLFIRewriter class, the X86 specific
// subclass of MCLFIRewriter.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_X86_MCTARGETDESC_X86MCLFIREWRITER_H
#define LLVM_LIB_TARGET_X86_MCTARGETDESC_X86MCLFIREWRITER_H

#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCLFIRewriter.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace llvm {
class MCContext;
class MCInst;
class MCStreamer;
class MCSubtargetInfo;

namespace X86 {

class X86MCLFIRewriter : public MCLFIRewriter {
public:
  X86MCLFIRewriter(MCContext &Ctx, std::unique_ptr<MCRegisterInfo> &&RI,
                   std::unique_ptr<MCInstrInfo> &&II)
      : MCLFIRewriter(Ctx, std::move(RI), std::move(II)) {}

  bool rewriteInst(const MCInst &Inst, MCStreamer &Out,
                   const MCSubtargetInfo &STI) override;

private:
  /// Recursion guard to prevent infinite loops when emitting instructions.
  bool Guard = false;

  void doRewriteInst(const MCInst &Inst, MCStreamer &Out,
                     const MCSubtargetInfo &STI);

  void rewriteSyscall(const MCInst &Inst, MCStreamer &Out,
                      const MCSubtargetInfo &STI);

  bool isFSAccess(const MCInst &Inst);
  void rewriteFSAccess(const MCInst &Inst, MCStreamer &Out,
                       const MCSubtargetInfo &STI);
};

} // namespace X86
} // namespace llvm
#endif // LLVM_LIB_TARGET_X86_MCTARGETDESC_X86MCLFIREWRITER_H
