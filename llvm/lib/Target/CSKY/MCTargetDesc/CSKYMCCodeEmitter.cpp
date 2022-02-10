//===-- CSKYMCCodeEmitter.cpp - CSKY Code Emitter interface ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CSKYMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#include "CSKYMCCodeEmitter.h"
#include "CSKYMCExpr.h"
#include "MCTargetDesc/CSKYMCTargetDesc.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;

#define DEBUG_TYPE "csky-mccode-emitter"

STATISTIC(MCNumEmitted, "Number of MC instructions emitted");

unsigned CSKYMCCodeEmitter::getOImmOpValue(const MCInst &MI, unsigned Idx,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(Idx);
  assert(MO.isImm() && "Unexpected MO type.");
  return MO.getImm() - 1;
}

unsigned
CSKYMCCodeEmitter::getImmOpValueIDLY(const MCInst &MI, unsigned Idx,
                                     SmallVectorImpl<MCFixup> &Fixups,
                                     const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(Idx);
  assert(MO.isImm() && "Unexpected MO type.");

  auto V = (MO.getImm() <= 3) ? 4 : MO.getImm();
  return V - 1;
}

unsigned
CSKYMCCodeEmitter::getImmOpValueMSBSize(const MCInst &MI, unsigned Idx,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        const MCSubtargetInfo &STI) const {
  const MCOperand &MSB = MI.getOperand(Idx);
  const MCOperand &LSB = MI.getOperand(Idx + 1);
  assert(MSB.isImm() && LSB.isImm() && "Unexpected MO type.");

  return MSB.getImm() - LSB.getImm();
}

static void writeData(uint32_t Bin, unsigned Size, raw_ostream &OS) {
  uint16_t LO16 = static_cast<uint16_t>(Bin);
  uint16_t HI16 = static_cast<uint16_t>(Bin >> 16);

  if (Size == 4)
    support::endian::write<uint16_t>(OS, HI16, support::little);

  support::endian::write<uint16_t>(OS, LO16, support::little);
}

void CSKYMCCodeEmitter::encodeInstruction(const MCInst &MI, raw_ostream &OS,
                                          SmallVectorImpl<MCFixup> &Fixups,
                                          const MCSubtargetInfo &STI) const {
  const MCInstrDesc &Desc = MII.get(MI.getOpcode());
  unsigned Size = Desc.getSize();

  ++MCNumEmitted;

  uint32_t Bin = getBinaryCodeForInstr(MI, Fixups, STI);

  uint16_t LO16 = static_cast<uint16_t>(Bin);
  uint16_t HI16 = static_cast<uint16_t>(Bin >> 16);

  if (Size == 4)
    support::endian::write<uint16_t>(OS, HI16, support::little);

  support::endian::write<uint16_t>(OS, LO16, support::little);
}

unsigned
CSKYMCCodeEmitter::getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                     SmallVectorImpl<MCFixup> &Fixups,
                                     const MCSubtargetInfo &STI) const {
  if (MO.isReg())
    return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());

  if (MO.isImm())
    return static_cast<unsigned>(MO.getImm());

  llvm_unreachable("Unhandled expression!");
  return 0;
}

unsigned
CSKYMCCodeEmitter::getRegSeqImmOpValue(const MCInst &MI, unsigned Idx,
                                       SmallVectorImpl<MCFixup> &Fixups,
                                       const MCSubtargetInfo &STI) const {
  assert(MI.getOperand(Idx).isReg() && "Unexpected MO type.");
  assert(MI.getOperand(Idx + 1).isImm() && "Unexpected MO type.");

  unsigned Ry = MI.getOperand(Idx).getReg();
  unsigned Rz = MI.getOperand(Idx + 1).getImm();

  unsigned Imm = Ctx.getRegisterInfo()->getEncodingValue(Rz) -
                 Ctx.getRegisterInfo()->getEncodingValue(Ry);

  return ((Ctx.getRegisterInfo()->getEncodingValue(Ry) << 5) | Imm);
}

unsigned
CSKYMCCodeEmitter::getRegisterSeqOpValue(const MCInst &MI, unsigned Op,
                                         SmallVectorImpl<MCFixup> &Fixups,
                                         const MCSubtargetInfo &STI) const {
  unsigned Reg1 =
      Ctx.getRegisterInfo()->getEncodingValue(MI.getOperand(Op).getReg());
  unsigned Reg2 =
      Ctx.getRegisterInfo()->getEncodingValue(MI.getOperand(Op + 1).getReg());

  unsigned Binary = ((Reg1 & 0x1f) << 5) | (Reg2 - Reg1);

  return Binary;
}

unsigned CSKYMCCodeEmitter::getImmJMPIX(const MCInst &MI, unsigned Idx,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        const MCSubtargetInfo &STI) const {
  if (MI.getOperand(Idx).getImm() == 16)
    return 0;
  else if (MI.getOperand(Idx).getImm() == 24)
    return 1;
  else if (MI.getOperand(Idx).getImm() == 32)
    return 2;
  else if (MI.getOperand(Idx).getImm() == 40)
    return 3;
  else
    assert(0);
}

MCFixupKind CSKYMCCodeEmitter::getTargetFixup(const MCExpr *Expr) const {
  const CSKYMCExpr *CSKYExpr = cast<CSKYMCExpr>(Expr);

  switch (CSKYExpr->getKind()) {
  default:
    llvm_unreachable("Unhandled fixup kind!");
  case CSKYMCExpr::VK_CSKY_ADDR:
    return MCFixupKind(CSKY::fixup_csky_addr32);
  case CSKYMCExpr::VK_CSKY_ADDR_HI16:
    return MCFixupKind(CSKY::fixup_csky_addr_hi16);
  case CSKYMCExpr::VK_CSKY_ADDR_LO16:
    return MCFixupKind(CSKY::fixup_csky_addr_lo16);
  case CSKYMCExpr::VK_CSKY_GOT:
    return MCFixupKind(CSKY::fixup_csky_got32);
  case CSKYMCExpr::VK_CSKY_GOTPC:
    return MCFixupKind(CSKY::fixup_csky_gotpc);
  case CSKYMCExpr::VK_CSKY_GOTOFF:
    return MCFixupKind(CSKY::fixup_csky_gotoff);
  case CSKYMCExpr::VK_CSKY_PLT:
    return MCFixupKind(CSKY::fixup_csky_plt32);
  case CSKYMCExpr::VK_CSKY_PLT_IMM18_BY4:
    return MCFixupKind(CSKY::fixup_csky_plt_imm18_scale4);
  case CSKYMCExpr::VK_CSKY_GOT_IMM18_BY4:
    return MCFixupKind(CSKY::fixup_csky_got_imm18_scale4);
  }
}

MCCodeEmitter *llvm::createCSKYMCCodeEmitter(const MCInstrInfo &MCII,
                                             const MCRegisterInfo &MRI,
                                             MCContext &Ctx) {
  return new CSKYMCCodeEmitter(Ctx, MCII);
}

#include "CSKYGenMCCodeEmitter.inc"
