//===-- DPUMCCodeEmitter.cpp - Convert DPU code to machine code -----------===//
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

#include "DPUMCCodeEmitter.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "mccodeemitter"

#define GET_INSTRINFO_ENUM
#include "DPUGenInstrInfo.inc"

#include "DPUAsmCondition.h"
#include "DPUFixupInfo.h"

#undef GET_INSTRINFO_ENUM

namespace llvm {

MCCodeEmitter *createDPUMCCodeEmitter(const MCInstrInfo &InstrInfo,
                                      const MCRegisterInfo &MRI,
                                      MCContext &context) {
  return new DPUMCCodeEmitter(InstrInfo, MRI, context);
}

void DPUMCCodeEmitter::encodeInstruction(const MCInst &MI, raw_ostream &OS,
                                         SmallVectorImpl<MCFixup> &Fixups,
                                         const MCSubtargetInfo &STI) const {
  // Get instruction encoding and emit it
  uint64_t Value = getBinaryCodeForInstr(MI, Fixups, STI);

  // Emit bytes in little-endian
  for (int i = 0; i < 8 * 8; i += 8) {
    OS << static_cast<char>((Value >> i) & 0xff);
  }
}

unsigned DPUMCCodeEmitter::getMachineOpValue(const MCInst &Inst,
                                             const MCOperand &MCOp,
                                             SmallVectorImpl<MCFixup> &Fixups,
                                             const MCSubtargetInfo &STI) const {
  if (MCOp.isReg()) {
    unsigned Reg = MCOp.getReg();
    unsigned RegNo = Ctx.getRegisterInfo()->getEncodingValue(Reg);
    return RegNo;
  }

  if (MCOp.isImm()) {
    return static_cast<unsigned>(MCOp.getImm());
  }

  // MO must be an Expr.
  assert(MCOp.isExpr());

  unsigned OpNo = static_cast<unsigned int>(&MCOp - Inst.begin());

  return getExprOpValue(MCOp.getExpr(), Inst.getOpcode(), OpNo, Fixups, STI);
}

unsigned DPUMCCodeEmitter::getExprOpValue(const MCExpr *Expr,
                                          unsigned InstOpcode, unsigned OpNum,
                                          SmallVectorImpl<MCFixup> &Fixups,
                                          const MCSubtargetInfo &STI) const {
  int64_t Res;

  if (Expr->evaluateAsAbsolute(Res)) {
    return static_cast<unsigned int>(Res);
  }

  DPU::Fixups FixupKind = DPU::findFixupForOperand(OpNum, InstOpcode);
  Fixups.push_back(MCFixup::create(0, Expr, MCFixupKind(FixupKind)));
  return 0;
}

unsigned
DPUMCCodeEmitter::getConditionEncoding(const MCInst &MI, unsigned OpNo,
                                       SmallVectorImpl<MCFixup> &Fixups,
                                       const MCSubtargetInfo &STI) const {
  assert(MI.getOperand(OpNo).isImm());
  DPUAsmCondition::Condition Cond =
      static_cast<DPUAsmCondition::Condition>(MI.getOperand(OpNo).getImm());
  unsigned int Opcode = MI.getOpcode();

  DPUAsmCondition::ConditionClass CondClass =
      DPUAsmCondition::findConditionClassForInstruction(Opcode);

  return static_cast<unsigned int>(
      DPUAsmCondition::getEncoding(Cond, CondClass));
}

#include "DPUGenMCCodeEmitter.inc"

} // namespace llvm
