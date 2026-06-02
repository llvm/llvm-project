//===-- EZHMCCodeEmitter.cpp - Convert EZH code to machine code -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the EZHMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/EZHBaseInfo.h"
#include "MCTargetDesc/EZHFixupKinds.h"
#include "MCTargetDesc/EZHMCTargetDesc.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "mccodeemitter"

namespace {
class EZHMCCodeEmitter : public MCCodeEmitter {
  const MCInstrInfo &MCII;
  MCContext &Ctx;

public:
  EZHMCCodeEmitter(const MCInstrInfo &mcii, MCContext &ctx)
      : MCII(mcii), Ctx(ctx) {}

  ~EZHMCCodeEmitter() override = default;

  void encodeInstruction(const MCInst &MI, SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  // getBinaryCodeForInstr - TableGen'erated function.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;

  // getMachineOpValue - Return binary encoding of operand.
  unsigned getWordOffsetOpValue(const MCInst &MI, unsigned OpNo,
                                SmallVectorImpl<MCFixup> &Fixups,
                                const MCSubtargetInfo &STI) const;

  unsigned getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  unsigned getBranchTargetOpValue(const MCInst &MI, unsigned OpNo,
                                  SmallVectorImpl<MCFixup> &Fixups,
                                  const MCSubtargetInfo &STI) const;

  unsigned getCallTargetOpValue(const MCInst &MI, unsigned OpNo,
                                SmallVectorImpl<MCFixup> &Fixups,
                                const MCSubtargetInfo &STI) const;
};
} // end anonymous namespace

void EZHMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                         SmallVectorImpl<char> &CB,
                                         SmallVectorImpl<MCFixup> &Fixups,
                                         const MCSubtargetInfo &STI) const {
  uint64_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);
  support::endian::write<uint32_t>(CB, Bits, llvm::endianness::little);
}

unsigned
EZHMCCodeEmitter::getBranchTargetOpValue(const MCInst &MI, unsigned OpNo,
                                         SmallVectorImpl<MCFixup> &Fixups,
                                         const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isReg())
    return getEZHRegisterNumbering(MO.getReg());
  if (MO.isImm()) {
    uint32_t val = static_cast<uint32_t>(MO.getImm());
    assert((val & 3) == 0 && "Branch target not 4-byte aligned!");
    return val >> 2;
  }

  Fixups.push_back(
      MCFixup::create(0, MO.getExpr(), MCFixupKind(EZH::FIXUP_EZH_21), false));
  return 0;
}

unsigned
EZHMCCodeEmitter::getCallTargetOpValue(const MCInst &MI, unsigned OpNo,
                                       SmallVectorImpl<MCFixup> &Fixups,
                                       const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isReg())
    return getEZHRegisterNumbering(MO.getReg());
  if (MO.isImm()) {
    uint32_t val = static_cast<uint32_t>(MO.getImm());
    assert((val & 3) == 0 && "Call offset not 4-byte aligned!");
    return val >> 2;
  }

  Fixups.push_back(
      MCFixup::create(0, MO.getExpr(), MCFixupKind(EZH::FIXUP_EZH_25)));
  return 0;
}

unsigned
EZHMCCodeEmitter::getWordOffsetOpValue(const MCInst &MI, unsigned OpNo,
                                       SmallVectorImpl<MCFixup> &Fixups,
                                       const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isImm()) {
    uint32_t val = static_cast<uint32_t>(MO.getImm());
    assert((val & 3) == 0 && "Word offset not 4-byte aligned!");
    return val >> 2;
  }

  if (MO.isExpr()) {
    Fixups.push_back(MCFixup::create(
        0, MO.getExpr(), MCFixupKind(EZH::FIXUP_EZH_8_PCREL), true));
    return 0;
  }

  return getMachineOpValue(MI, MO, Fixups, STI);
}

unsigned EZHMCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                             const MCOperand &MO,
                                             SmallVectorImpl<MCFixup> &Fixups,
                                             const MCSubtargetInfo &STI) const {
  if (MO.isReg())
    return getEZHRegisterNumbering(MO.getReg());
  if (MO.isImm()) {
    int64_t Imm = MO.getImm();
    unsigned Opc = MI.getOpcode();

    if (Opc == EZH::MOVri__) {
      if (!isInt<11>(Imm)) {
        Ctx.reportError(MI.getLoc(),
                        "immediate operand " + Twine(Imm) +
                            " is out of range for e_load_imm (requires 11-bit "
                            "signed immediate, -1024 to 1023)!");
        return 0;
      }
    } else if (Opc == EZH::ADDri__ || Opc == EZH::SUBri__) {
      if (!isInt<12>(Imm)) {
        Ctx.reportError(MI.getLoc(),
                        "immediate operand " + Twine(Imm) +
                            " is out of range for e_add/sub_imm (requires "
                            "12-bit signed immediate, -2048 to 2047)!");
        return 0;
      }
    } else if (Opc == EZH::LSLi__ || Opc == EZH::LSRi__ || Opc == EZH::ASRi__ ||
               Opc == EZH::RORi__) {
      if (!isUInt<5>(Imm)) {
        Ctx.reportError(MI.getLoc(), "shift count immediate " + Twine(Imm) +
                                         " is out of range (requires 5-bit "
                                         "unsigned immediate, 0 to 31)!");
        return 0;
      }
    }

    return static_cast<unsigned>(Imm);
  }

  if (MO.isExpr()) {
    unsigned Opc = MI.getOpcode();
    unsigned FixupKind = EZH::FIXUP_EZH_32;
    if (Opc == EZH::MOVSri__)
      FixupKind = EZH::FIXUP_EZH_HI16;
    else if (Opc == EZH::ORri__)
      FixupKind = EZH::FIXUP_EZH_LO16;

    Fixups.push_back(MCFixup::create(0, MO.getExpr(), MCFixupKind(FixupKind)));
    return 0;
  }

  return 0;
}

MCCodeEmitter *llvm::createEZHMCCodeEmitter(const MCInstrInfo &MCII,
                                            MCContext &Ctx) {
  return new EZHMCCodeEmitter(MCII, Ctx);
}

#include "EZHGenMCCodeEmitter.inc"
