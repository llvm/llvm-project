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


#include "SuperHMCTargetDesc.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/bit.h"
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


using namespace llvm;

#define DEBUG_TYPE "mccodeemitter"

STATISTIC(MCNumEmitted, "Number of MC instructions emitted");

namespace {

class SuperHMCCodeEmitter : public MCCodeEmitter {
  MCContext &Ctx;

public:
  SuperHMCCodeEmitter(const MCInstrInfo &, MCContext &ctx)
    : Ctx(ctx) {}
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
};

} // end namespace

#include "SuperHGenMCCodeEmitter.inc"

void SuperHMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                           SmallVectorImpl<char> &CB,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
  
  // All base instructions are 16-bit in SuperH asm
  uint16_t Bits = (uint16_t)getBinaryCodeForInstr(MI, Fixups, STI);
  support::endian::write(CB, Bits, Ctx.getAsmInfo().isLittleEndian()
                                      ? llvm::endianness::little
                                      : llvm::endianness::big);

  ++MCNumEmitted;
}

unsigned SuperHMCCodeEmitter::getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const {
  if (MO.isReg())
      return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());

  if (MO.isImm())
    return MO.getImm();

  return 0;
}

MCCodeEmitter *llvm::createSuperHMCCodeEmitter(const MCInstrInfo &MCII,
                                              MCContext &Ctx) {
  return new SuperHMCCodeEmitter(MCII, Ctx);
}