//===-- DPUMCCodeEmitter.h - Convert DPU code to machine code -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the DPUMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DPU_MCTARGETDESC_DPUMCCODEEMITTER_H
#define LLVM_LIB_TARGET_DPU_MCTARGETDESC_DPUMCCODEEMITTER_H

#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include <cstdint>

namespace llvm {
class DPUMCCodeEmitter : public MCCodeEmitter {
  const MCInstrInfo &MCII;
  const MCRegisterInfo &MRI;
  MCContext &Ctx;

public:
  DPUMCCodeEmitter(const MCInstrInfo &mcii, const MCRegisterInfo &mri,
                   MCContext &ctx)
      : MCII(mcii), MRI(mri), Ctx(ctx) {}

  void encodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  // getBinaryCodeForInstr - TableGen'erated function for getting the
  // binary encoding for an instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;

  // getMachineOpValue - Return binary encoding of operand. If the machine
  // operand requires relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &Inst, const MCOperand &MCOp,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &SubtargetInfo) const;

  unsigned getConditionEncoding(const MCInst &MI, unsigned OpNo,
                                SmallVectorImpl<MCFixup> &Fixups,
                                const MCSubtargetInfo &STI) const;

private:
  unsigned getExprOpValue(const MCExpr *Expr, unsigned InstOpcode,
                          unsigned OpNum, SmallVectorImpl<MCFixup> &Fixups,
                          const MCSubtargetInfo &STI) const;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_DPU_MCTARGETDESC_DPUMCCODEEMITTER_H
