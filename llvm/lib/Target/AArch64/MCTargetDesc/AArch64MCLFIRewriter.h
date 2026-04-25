//===- AArch64MCLFIRewriter.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AArch64MCLFIRewriter class, the AArch64 specific
// subclass of MCLFIRewriter.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_AARCH64_MCTARGETDESC_AARCH64MCLFIREWRITER_H
#define LLVM_LIB_TARGET_AARCH64_MCTARGETDESC_AARCH64MCLFIREWRITER_H

#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCLFIRewriter.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace llvm {
class MCContext;
class MCInst;
class MCStreamer;
class MCSubtargetInfo;

/// Rewrites AArch64 instructions for LFI sandboxing.
///
/// This class implements the LFI (Lightweight Fault Isolation) rewriting
/// for AArch64 instructions. It transforms instructions to ensure memory
/// accesses and control flow are confined within the sandbox region.
///
/// Reserved registers:
/// - X27: Sandbox base address (always holds the base)
/// - X28: Safe address register (always within sandbox)
/// - X26: Scratch register for intermediate calculations
/// - X25: context register (points to thread-local runtime data)
/// - SP:  Stack pointer (always within sandbox)
/// - X30: Link register (always within sandbox)
class AArch64MCLFIRewriter : public MCLFIRewriter {
public:
  AArch64MCLFIRewriter(MCContext &Ctx, std::unique_ptr<MCRegisterInfo> &&RI,
                       std::unique_ptr<MCInstrInfo> &&II)
      : MCLFIRewriter(Ctx, std::move(RI), std::move(II)) {}

  bool rewriteInst(const MCInst &Inst, MCStreamer &Out,
                   const MCSubtargetInfo &STI) override;

private:
  /// Recursion guard to prevent infinite loops when emitting instructions.
  bool Guard = false;

  // Instruction classification.
  bool mayModifyReserved(const MCInst &Inst) const;

  // Instruction emission.
  void emitInst(const MCInst &Inst, MCStreamer &Out,
                const MCSubtargetInfo &STI);
  void emitAddMask(MCRegister Dest, MCRegister Src, MCStreamer &Out,
                   const MCSubtargetInfo &STI);
  void emitBranch(unsigned Opcode, MCRegister Target, MCStreamer &Out,
                  const MCSubtargetInfo &STI);
  void emitMov(MCRegister Dest, MCRegister Src, MCStreamer &Out,
               const MCSubtargetInfo &STI);

  // Rewriting logic.
  void doRewriteInst(const MCInst &Inst, MCStreamer &Out,
                     const MCSubtargetInfo &STI);

  // Control flow.
  void rewriteIndirectBranch(const MCInst &Inst, MCStreamer &Out,
                             const MCSubtargetInfo &STI);
  void rewriteReturn(const MCInst &Inst, MCStreamer &Out,
                     const MCSubtargetInfo &STI);

  // Link register modification.
  void rewriteLRModification(const MCInst &Inst, MCStreamer &Out,
                             const MCSubtargetInfo &STI);

  // System instructions.
  void rewriteSyscall(const MCInst &Inst, MCStreamer &Out,
                      const MCSubtargetInfo &STI);
  void rewriteTPRead(const MCInst &Inst, MCStreamer &Out,
                     const MCSubtargetInfo &STI);
  void rewriteTPWrite(const MCInst &Inst, MCStreamer &Out,
                      const MCSubtargetInfo &STI);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AARCH64_MCTARGETDESC_AARCH64MCLFIREWRITER_H
