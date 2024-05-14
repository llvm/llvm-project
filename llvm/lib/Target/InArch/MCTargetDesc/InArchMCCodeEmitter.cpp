//===-- InArchMCCodeEmitter.cpp - Convert InArch code to machine code -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the InArchMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#include "InArch.h"
#include "MCTargetDesc/InArchMCTargetDesc.h"
#include "MCTargetDesc/InArchFixupKinds.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
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
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "mccodeemitter"

STATISTIC(MCNumEmitted, "Number of MC instructions emitted");

namespace {

class InArchMCCodeEmitter : public MCCodeEmitter {
  MCContext &Ctx;

public:
  InArchMCCodeEmitter(const MCInstrInfo &, MCContext &ctx)
      : Ctx(ctx) {}
  InArchMCCodeEmitter(const InArchMCCodeEmitter &) = delete;
  InArchMCCodeEmitter &operator=(const InArchMCCodeEmitter &) = delete;
  ~InArchMCCodeEmitter() override = default;

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
  unsigned getSImm16OpValue(const MCInst &MI, unsigned OpNo,
                            SmallVectorImpl<MCFixup> &Fixups,
                            const MCSubtargetInfo &STI) const;
  unsigned getBranchTarget16OpValue(const MCInst &MI, unsigned OpNo,
                            SmallVectorImpl<MCFixup> &Fixups,
                            const MCSubtargetInfo &STI) const;
};

} // end anonymous namespace

void InArchMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                           SmallVectorImpl<char> &CB,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
  unsigned Bits = getBinaryCodeForInstr(MI, Fixups, STI);
  support::endian::write(CB, Bits, llvm::endianness::little);

  ++MCNumEmitted;  // Keep track of the # of mi's emitted.
}

unsigned InArchMCCodeEmitter::
getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                  SmallVectorImpl<MCFixup> &Fixups,
                  const MCSubtargetInfo &STI) const {
  if (MO.isReg())
    return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());

  if (MO.isImm())
    return MO.getImm();

  assert(MO.isExpr());
  const MCExpr *Expr = MO.getExpr();

  int64_t Res;
  if (Expr->evaluateAsAbsolute(Res))
    return Res;

  llvm_unreachable("Unhandled expression!");
  return 0;
}

unsigned
InArchMCCodeEmitter::getSImm16OpValue(const MCInst &MI, unsigned OpNo,
                                     SmallVectorImpl<MCFixup> &Fixups,
                                     const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);
  if (MO.isImm())
    return MO.getImm();

  assert(MO.isExpr() &&
         "getSImm16OpValue expects only expressions or an immediate");

  const MCExpr *Expr = MO.getExpr();

  // Constant value, no fixup is needed
  if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
    return CE->getValue();

  return 0;
}

/// getBranchTarget21OpValue - Return binary encoding of the branch
/// target operand. If the machine operand requires relocation,
/// record the relocation and return zero.
unsigned InArchMCCodeEmitter::
getBranchTarget16OpValue(const MCInst &MI, unsigned OpNo,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const {
  const MCOperand &MO = MI.getOperand(OpNo);

  // If the destination is an immediate, divide by 4.
  if (MO.isImm()) return MO.getImm() / 4;

  assert(MO.isExpr() &&
         "getBranchTarget16OpValue expects only expressions or immediates");

  Fixups.push_back(MCFixup::create(0, MO.getExpr(),
                                   MCFixupKind(InArch::fixup_InArch_PC16)));
  return 0;
}

#include "InArchGenMCCodeEmitter.inc"

MCCodeEmitter *llvm::createInArchMCCodeEmitter(const MCInstrInfo &MCII,
                                              MCContext &Ctx) {
  return new InArchMCCodeEmitter(MCII, Ctx);
}