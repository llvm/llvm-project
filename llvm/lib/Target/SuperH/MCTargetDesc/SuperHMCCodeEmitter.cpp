//===-- SuperHGenMCCodeEmitter.cpp - Convert SuperH code to machine code --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SuperHMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#include "SuperHFixupKinds.h"
#include "SuperHInstrInfo.h"
#include "SuperHMCTargetDesc.h"
#include "SuperHSubtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "sh-mccodeemitter"

STATISTIC(MCNumEmitted, "Number of MC instructions emitted");

namespace {

class SuperHMCCodeEmitter : public MCCodeEmitter {
  MCContext &Ctx;
  const SuperHInstrInfo *TII;

public:
  SuperHMCCodeEmitter(const MCInstrInfo &II, MCContext &ctx) : 
    Ctx(ctx),
    TII((const SuperHInstrInfo *)&II) {}
  SuperHMCCodeEmitter(const SuperHMCCodeEmitter &) = delete;
  SuperHMCCodeEmitter &operator=(const SuperHMCCodeEmitter &) = delete;
  ~SuperHMCCodeEmitter() override = default;

  void encodeInstruction(const MCInst &MI, SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  // getBinaryCodeForInstr - TableGen'erated function for getting the
  // binary encoding for an instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;

  /// getMachineOpValue - Return binary encoding of operand. If the machine
  /// operand requires relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  unsigned getExprOpValue(const MCInst &MI, const MCExpr *Expr,
                          SmallVectorImpl<MCFixup> &Fixups,
                          const MCSubtargetInfo &STI, int64_t Shift) const;

  unsigned getBranchTargetOpValue(const MCInst &MI, unsigned OpNo,
                                  SmallVectorImpl<MCFixup> &Fixups,
                                  const MCSubtargetInfo &STI) const;
  
  MCFixup getFixupForOpcode(const MCInst &MI, const MCExpr *Expr,
                                 MCContext &Ctx) const;

  // Displacement
  template <int Scale>
  unsigned getDispOpValue(const MCInst &MI, unsigned OpNo,
                          SmallVectorImpl<MCFixup> &Fixups,
                          const MCSubtargetInfo &STI) const;

  // PC-relative displacement.
  template <int Scale>
  unsigned getPCRelOpValue(const MCInst &MI, unsigned OpNo,
                           SmallVectorImpl<MCFixup> &Fixups,
                           const MCSubtargetInfo &STI) const;
};

} // end namespace

#include "SuperHGenMCCodeEmitter.inc"

// @getFixupForOpcode - Helper that gets the neccesary fixup for
// the given opcode.
// This is neccesary due to the various opcodes that access memory
// have different scaling factors applied.
MCFixup SuperHMCCodeEmitter::getFixupForOpcode(const MCInst &MI, const MCExpr *Expr,
                                               MCContext &Ctx) const {
  int Size = TII->getInstSizeInBytes(MI);
  MCFixupKind Kind = Size == 4 ? FK_Data_4 : FK_Data_2;

  switch (MI.getOpcode()) {
  default:
    break;

  // disp4 * 2
  case SH::MOVWL4:
  case SH::MOVWLG:
  case SH::MOVWS4:
  case SH::MOVWSG:
    Kind = SH::fixup_pcrel4_by2;
    break;

  // disp4 * 4
  case SH::MOVLLG:
  case SH::MOVLS4:
  case SH::MOVLSG:
    Kind = SH::fixup_pcrel4_by4;
    break;

  // disp8 * 4
  case SH::MOVA:
    Kind = SH::fixup_pcrel8_by4;
    break;

  // (disp8 * 2) + 4
  case SH::BF:
  case SH::BFS:
  case SH::BT:
  case SH::BTS:
  case SH::MOVWI:
  case SH::MOVLL4:
    Kind = SH::fixup_pcrel8_4by2;
    break;

  // (disp8 * 4) + 4
  case SH::MOVLI:
    Kind = SH::fixup_pcrel8_4by4;
    break;

  // (disp12 * 2) + 4
  case SH::BSR:
  case SH::BRA:
    Kind = SH::fixup_pcrel12_4by2;
    break;
  }

  return MCFixup::create(0, Expr, Kind, true);
}

void SuperHMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                            SmallVectorImpl<char> &CB,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
  auto Size = TII->getInstSizeInBytes(MI);
  uint64_t OpCode = getBinaryCodeForInstr(MI, Fixups, STI);

  // Fill buffer of data to insert to the instruction stream.
  bool SwapValue = !Ctx.getAsmInfo().isLittleEndian();
  for (unsigned i = 0; i < Size; ++i) {
    unsigned Idx = SwapValue ? (Size - 1 - i) : i;
    CB.push_back(uint8_t(((OpCode >> (Idx * 8)) & 0xff)));
  }

  ++MCNumEmitted;
}




//===----------------------------------------------------------------------===//
//                              Branch Target
//===----------------------------------------------------------------------===//

unsigned
SuperHMCCodeEmitter::getBranchTargetOpValue(const MCInst &MI, unsigned OpNo,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
  return getMachineOpValue(MI, MI.getOperand(OpNo), Fixups, STI);
}




//===----------------------------------------------------------------------===//
//                                Displacement
//===----------------------------------------------------------------------===//

template <int Scale>
unsigned SuperHMCCodeEmitter::getDispOpValue(const MCInst &MI, unsigned OpNo,
                                             SmallVectorImpl<MCFixup> &Fixups,
                                             const MCSubtargetInfo &STI) const {
  // Skip base register if found.
  if (MI.getOperand(OpNo).isReg())
    OpNo++;

  auto MO = MI.getOperand(OpNo);
  if (MO.isImm())
    return (MO.getImm() / Scale);

  assert(MO.isExpr() && "Expected Expression");
  return getMachineOpValue(MI, MI.getOperand(OpNo), Fixups, STI);
}





//===----------------------------------------------------------------------===//
//                          PC-Relative Displacement
//===----------------------------------------------------------------------===//

template <int Scale>
unsigned
SuperHMCCodeEmitter::getPCRelOpValue(const MCInst &MI, unsigned OpNo,
                                     SmallVectorImpl<MCFixup> &Fixups,
                                     const MCSubtargetInfo &STI) const {
  auto MO = MI.getOperand(OpNo);
  if (MO.isImm())
    return MO.getImm() / Scale;

  assert(MO.isExpr() && "Expected Expression");
  return getExprOpValue(MI, MO.getExpr(), Fixups, STI, Scale);
}

unsigned SuperHMCCodeEmitter::getExprOpValue(const MCInst &MI,
                                             const MCExpr *Expr,
                                             SmallVectorImpl<MCFixup> &Fixups,
                                             const MCSubtargetInfo &STI,
                                             int64_t Shift) const {
  if (!Expr)
    return 0;

  switch (Expr->getKind()) {
  case MCExpr::ExprKind::Binary: {
    unsigned Res =
        getExprOpValue(MI, static_cast<const MCBinaryExpr *>(Expr)->getLHS(),
                       Fixups, STI, Shift);
    Res += getExprOpValue(MI, static_cast<const MCBinaryExpr *>(Expr)->getRHS(),
                          Fixups, STI, Shift);
    return Res;
  }
  case MCExpr::ExprKind::Target: {
    llvm_unreachable("TODO");
  }
  case MCExpr::ExprKind::Specifier: {
    const MCSpecifierExpr *Spec = static_cast<const MCSpecifierExpr *>(Expr);
    Fixups.push_back(MCFixup::create(0, Spec, Spec->getSpecifier()));
    return 0;
  }
  case MCExpr::ExprKind::SymbolRef: {
    Fixups.push_back(getFixupForOpcode(MI, Expr, Ctx));
    return 0;
  }
  case MCExpr::ExprKind::Constant: {
    const MCConstantExpr *Const = static_cast<const MCConstantExpr *>(Expr);
    return Const->getValue();
  }
  default:
    llvm_unreachable("expression not supported!");
  }
}

/// getMachineOpValue - Return binary encoding of operand. If the machine
/// operand requires relocation, record the relocation and return zero.
unsigned
SuperHMCCodeEmitter::getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                       SmallVectorImpl<MCFixup> &Fixups,
                                       const MCSubtargetInfo &STI) const {
  if (MO.isReg())
    return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());

  if (MO.isImm())
    return MO.getImm();

  assert(MO.isExpr() && "Expected Expression");
  return getExprOpValue(MI, MO.getExpr(), Fixups, STI, 0);
}

MCCodeEmitter *llvm::createSuperHMCCodeEmitter(const MCInstrInfo &MCII,
                                               MCContext &Ctx) {
  return new SuperHMCCodeEmitter(MCII, Ctx);
}