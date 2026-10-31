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

#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCLFIRewriter.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace llvm {
class MCContext;
class MCStreamer;
class MCSubtargetInfo;

namespace X86 {

constexpr unsigned LFIBundleSize = 32;

class X86MCLFIRewriter : public MCLFIRewriter {
public:
  X86MCLFIRewriter(MCContext &Ctx, std::unique_ptr<MCRegisterInfo> &&RI,
                   std::unique_ptr<MCInstrInfo> &&II)
      : MCLFIRewriter(Ctx, std::move(RI), std::move(II)) {}

  bool rewriteInst(const MCInst &Inst, MCStreamer &Out,
                   const MCSubtargetInfo &STI) override;

  void onLabel(const MCSymbol *Symbol, MCStreamer &Out) override;
  void finish(MCStreamer &Out) override;

private:
  /// Recursion guard to prevent infinite loops when emitting instructions.
  bool Guard = false;

  /// Prefixes written on their own line, held back so that they can be emitted
  /// immediately before the instruction they apply to.
  SmallVector<MCInst, 2> Prefixes;

  void doRewriteInst(const MCInst &Inst, MCStreamer &Out,
                     const MCSubtargetInfo &STI);

  /// Emit the prefixes that were held back, followed by Inst.
  void emitWithPrefixes(const MCInst &Inst, MCStreamer &Out,
                        const MCSubtargetInfo &STI);

  /// Report and drop the prefixes that were held back, for when the
  /// instruction they would apply to does not take them.
  void discardPrefixes();

  /// Return the 32-bit subregister of Reg, or Reg itself if it already
  /// is a 32-bit register.
  MCRegister getReg32(MCRegister Reg) const;

  void rewriteSyscall(const MCInst &Inst, MCStreamer &Out,
                      const MCSubtargetInfo &STI);

  /// Emit the mask sequence that turns an arbitrary value in Reg into a
  /// bundle-aligned address inside the sandbox.
  void emitSandboxBranchReg(MCRegister Reg, MCStreamer &Out,
                            const MCSubtargetInfo &STI);

  void rewriteIndirectBranch(const MCInst &Inst, MCStreamer &Out,
                             const MCSubtargetInfo &STI);
  void rewriteDirectCall(const MCInst &Inst, MCStreamer &Out,
                         const MCSubtargetInfo &STI);
  void rewriteReturn(const MCInst &Inst, MCStreamer &Out,
                     const MCSubtargetInfo &STI);

  bool isFSAccess(const MCInst &Inst);
  void rewriteFSAccess(const MCInst &Inst, MCStreamer &Out,
                       const MCSubtargetInfo &STI);

  /// Rewrite the addressing mode of the memory operand at MemIdx so that
  /// the access it performs stays inside the sandbox.
  void sandboxMemOperand(MCInst &Inst, int MemIdx) const;

  /// Apply sandboxMemOperand to the memory operand of Inst, if it has one
  /// that it dereferences.
  void sandboxMemAccess(MCInst &Inst) const;

  void rewriteMemAccess(const MCInst &Inst, MCStreamer &Out,
                        const MCSubtargetInfo &STI);
  void rewriteStringOperation(const MCInst &Inst, MCStreamer &Out,
                              const MCSubtargetInfo &STI);

  /// Return the stack pointer register that Inst writes as an explicit
  /// operand, or NoRegister if it does not write one.
  MCRegister getWrittenStackReg(const MCInst &Inst) const;

  void rewriteStackModification(MCRegister StackReg, const MCInst &Inst,
                                MCStreamer &Out, const MCSubtargetInfo &STI);
  void rewriteLeave(const MCInst &Inst, MCStreamer &Out,
                    const MCSubtargetInfo &STI);
};

} // namespace X86
} // namespace llvm
#endif // LLVM_LIB_TARGET_X86_MCTARGETDESC_X86MCLFIREWRITER_H
