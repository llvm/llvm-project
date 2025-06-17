//===-- SparcMCCodeEmitter.cpp - Convert Sparc code to machine code -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SparcMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SparcFixupKinds.h"
#include "SparcMCExpr.h"
#include "SparcMCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "mccodeemitter"

STATISTIC(MCNumEmitted, "Number of MC instructions emitted");

namespace {

class SparcMCCodeEmitter : public MCCodeEmitter {
  MCContext &Ctx;

public:
  SparcMCCodeEmitter(const MCInstrInfo &, MCContext &ctx)
      : Ctx(ctx) {}
  SparcMCCodeEmitter(const SparcMCCodeEmitter &) = delete;
  SparcMCCodeEmitter &operator=(const SparcMCCodeEmitter &) = delete;
  ~SparcMCCodeEmitter() override = default;

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
  unsigned getCallTargetOpValue(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;
  unsigned getBranchTargetOpValue(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;
  unsigned getSImm5OpValue(const MCInst &MI, unsigned OpNo,
                           SmallVectorImpl<MCFixup> &Fixups,
                           const MCSubtargetInfo &STI) const;
  unsigned getSImm13OpValue(const MCInst &MI, unsigned OpNo,
                            SmallVectorImpl<MCFixup> &Fixups,
                            const MCSubtargetInfo &STI) const;
  unsigned getBranchPredTargetOpValue(const MCInst &MI, unsigned OpNo,
                                      SmallVectorImpl<MCFixup> &Fixups,
                                      const MCSubtargetInfo &STI) const;
  unsigned getBranchOnRegTargetOpValue(const MCInst &MI, unsigned OpNo,
                                       SmallVectorImpl<MCFixup> &Fixups,
                                       const MCSubtargetInfo &STI) const;
  unsigned getCompareAndBranchTargetOpValue(const MCInst &MI, unsigned OpNo,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const;
};

} // end anonymous namespace

void SparcMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                           SmallVectorImpl<char> &CB,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
  unsigned Bits = getBinaryCodeForInstr(MI, Fixups, STI);
  support::endian::write(CB, Bits,
                         Ctx.getAsmInfo()->isLittleEndian()
                             ? llvm::endianness::little
                             : llvm::endianness::big);

  // Some instructions have phantom operands that only contribute a fixup entry.
  unsigned SymOpNo = 0;
  switch (MI.getOpcode()) {
  default: break;
  case SP::TLS_CALL:   SymOpNo = 1; break;
  case SP::GDOP_LDrr:
  case SP::GDOP_LDXrr:
  case SP::TLS_ADDrr:
  case SP::TLS_LDrr:
  case SP::TLS_LDXrr:  SymOpNo = 3; break;
  }
  if (SymOpNo != 0) {
    const MCOperand &MO = MI.getOperand(SymOpNo);
    uint64_t op = getMachineOpValue(MI, MO, Fixups, STI);
    assert(op == 0 && "Unexpected operand value!");
    (void)op; // suppress warning.
  }

  ++MCNumEmitted;  // Keep track of the # of mi's emitted.
}

unsigned SparcMCCodeEmitter::
getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                  SmallVectorImpl<MCFixup> &Fixups,
                  const MCSubtargetInfo &STI) const {
  if (MO.isReg())
    return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());

  if (MO.isImm())
    return MO.getImm();

  assert(MO.isExpr());
  const MCExpr *Expr = MO.getExpr();
  if (const SparcMCExpr *SExpr = dyn_cast<SparcMCExpr>(Expr)) {
    Fixups.push_back(MCFixup::create(0, Expr, SExpr->getSpecifier()));
    return 0;
  }

  int64_t Res;
  if (Expr->evaluateAsAbsolute(Res))
    return Res;

  llvm_unreachable("Unhandled expression!");
  return 0;
}

unsigned SparcMCCodeEmitter::getSImm5OpValue(const MCInst &MI, unsigned OpNo,
                                             SmallVectorImpl<MCFixup> &Fixups,
                                             const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isImm())
    return MO.getImm();

  assert(MO.isExpr() &&
         "getSImm5OpValue expects only expressions or an immediate");

  const MCExpr *Expr = MO.getExpr();

  // Constant value, no fixup is needed
  if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
    return CE->getValue();

  if (const SparcMCExpr *SExpr = dyn_cast<SparcMCExpr>(Expr)) {
    Fixups.push_back(MCFixup::create(0, Expr, SExpr->getSpecifier()));
    return 0;
  }
  Fixups.push_back(MCFixup::create(0, Expr, ELF::R_SPARC_5));
  return 0;
}

unsigned
SparcMCCodeEmitter::getSImm13OpValue(const MCInst &MI, unsigned OpNo,
                                     SmallVectorImpl<MCFixup> &Fixups,
                                     const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);

  if (MO.isImm())
    return MO.getImm();

  assert(MO.isExpr() &&
         "getSImm13OpValue expects only expressions or an immediate");

  const MCExpr *Expr = MO.getExpr();

  // Constant value, no fixup is needed
  if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
    return CE->getValue();

  if (const SparcMCExpr *SExpr = dyn_cast<SparcMCExpr>(Expr)) {
    Fixups.push_back(MCFixup::create(0, Expr, SExpr->getSpecifier()));
    return 0;
  }
  Fixups.push_back(MCFixup::create(0, Expr, Sparc::fixup_sparc_13));
  return 0;
}

unsigned SparcMCCodeEmitter::
getCallTargetOpValue(const MCInst &MI, unsigned OpNo,
                     SmallVectorImpl<MCFixup> &Fixups,
                     const MCSubtargetInfo &STI) const {
  if (MI.getOpcode() == SP::TLS_CALL) {
    // No fixups for __tls_get_addr. Will emit for fixups for tls_symbol in
    // encodeInstruction.
    return 0;
  }

  const MCOperand &MO = MI.getOperand(OpNo);
  Fixups.push_back(MCFixup::create(0, MO.getExpr(), Sparc::fixup_sparc_call30));
  return 0;
}

unsigned SparcMCCodeEmitter::
getBranchTargetOpValue(const MCInst &MI, unsigned OpNo,
                  SmallVectorImpl<MCFixup> &Fixups,
                  const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isReg() || MO.isImm())
    return getMachineOpValue(MI, MO, Fixups, STI);

  Fixups.push_back(MCFixup::create(0, MO.getExpr(), ELF::R_SPARC_WDISP22));
  return 0;
}

unsigned SparcMCCodeEmitter::getBranchPredTargetOpValue(
    const MCInst &MI, unsigned OpNo, SmallVectorImpl<MCFixup> &Fixups,
    const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isReg() || MO.isImm())
    return getMachineOpValue(MI, MO, Fixups, STI);

  Fixups.push_back(MCFixup::create(0, MO.getExpr(), ELF::R_SPARC_WDISP19));
  return 0;
}

unsigned SparcMCCodeEmitter::getBranchOnRegTargetOpValue(
    const MCInst &MI, unsigned OpNo, SmallVectorImpl<MCFixup> &Fixups,
    const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isReg() || MO.isImm())
    return getMachineOpValue(MI, MO, Fixups, STI);

  Fixups.push_back(MCFixup::create(0, MO.getExpr(), ELF::R_SPARC_WDISP16));

  return 0;
}

unsigned SparcMCCodeEmitter::getCompareAndBranchTargetOpValue(
    const MCInst &MI, unsigned OpNo, SmallVectorImpl<MCFixup> &Fixups,
    const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isImm())
    return getMachineOpValue(MI, MO, Fixups, STI);

  Fixups.push_back(MCFixup::create(0, MO.getExpr(), ELF::R_SPARC_WDISP10));

  return 0;
}

#include "SparcGenMCCodeEmitter.inc"

MCCodeEmitter *llvm::createSparcMCCodeEmitter(const MCInstrInfo &MCII,
                                              MCContext &Ctx) {
  return new SparcMCCodeEmitter(MCII, Ctx);
}
