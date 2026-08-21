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
#include "SuperHMCAsmInfo.h"
#include "SuperHMCTargetDesc.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/MachineInstr.h"
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
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/EndianStream.h"
#include <cstdint>


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

  unsigned getBranchTargetOpValue(const MCInst &MI, const MCExpr *Expr,
                          SmallVectorImpl<MCFixup> &Fixups,
                          const MCSubtargetInfo &STI) const;

  unsigned getExprOpValue(const MCInst &MI, const MCExpr *Expr,
                          SmallVectorImpl<MCFixup> &Fixups,
                          const MCSubtargetInfo &STI) const;

  unsigned getOpBits(const MCInst &MI,
                     SmallVectorImpl<MCFixup> &Fixups,
                     const MCSubtargetInfo &STI) const;
};

} // end namespace

#include "SuperHGenMCCodeEmitter.inc"

// Some SuperH instructions are 32-bits wide.
//
// Checks the passed opcode for any bit patterns
// that must be encoded as 32-bits.
static bool isOpcode32(uint32_t Opcode) {
  
  // movi20 & movi20s
  if ((Opcode & 0xF00F) <= 0x0001)
    return true;
  
  // Other SH2A 32-bit instructions
  if ((Opcode & 0xF00F) == 0x3001)
    return true;

  // Opcodes bigger than 0xFFFF are always 32-bits.
  return Opcode > 0xFFFF; 
}

// Helper that gets the bits for the given instruction.
unsigned SuperHMCCodeEmitter::getOpBits(const MCInst &MI,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        const MCSubtargetInfo &STI) const {
  MCInst Inst = MCInst();
  Inst.setOpcode(MI.getOpcode());
  for(unsigned i = 0; i < MI.getNumOperands(); i++) {
    Inst.addOperand(MCOperand::createImm(0));
  }

  return getBinaryCodeForInstr(Inst, Fixups, STI);
}

void SuperHMCCodeEmitter::encodeInstruction(const MCInst &MI,
                                           SmallVectorImpl<char> &CB,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {

  uint64_t OpCode = getBinaryCodeForInstr(MI, Fixups, STI);

  // NOTE:  All base instructions are 16-bit in SH ASM
  //        But some instructions may be 32-bit for eg. SH2A or the DSP extensions.
  //        This is ugly, but it'll work.
  if (isOpcode32(OpCode)) {
    support::endian::write(CB, (uint32_t)OpCode, Ctx.getAsmInfo().isLittleEndian()
                                        ? llvm::endianness::little
                                        : llvm::endianness::big);
  } else {
    support::endian::write(CB, (uint16_t)OpCode, Ctx.getAsmInfo().isLittleEndian()
                                        ? llvm::endianness::little
                                        : llvm::endianness::big);

  }
  ++MCNumEmitted;
}

unsigned SuperHMCCodeEmitter::getBranchTargetOpValue(const MCInst &MI, 
                                                     const MCExpr *Expr,
                                                     SmallVectorImpl<MCFixup> &Fixups,
                                                     const MCSubtargetInfo &STI) const {
  return getExprOpValue(MI, Expr, Fixups, STI);
}

unsigned SuperHMCCodeEmitter::getExprOpValue(const MCInst &MI, const MCExpr *Expr,
                                             SmallVectorImpl<MCFixup> &Fixups,
                                             const MCSubtargetInfo &STI) const {
  if (!Expr)
    return 0;

  MCExpr::ExprKind Kind = Expr->getKind();

  // Binary Op
  if (Kind == MCExpr::Binary) {
    Expr = static_cast<const MCBinaryExpr *>(Expr)->getLHS();
    Kind = Expr->getKind();
  }

  if (Kind == MCExpr::SymbolRef) {

    // NOTE:  A few (DSP and SH2A) instructions are 32-bits wide.
    //        We handle those quite crudely.
    uint32_t OpCode = getOpBits(MI, Fixups, STI);
    Fixups.push_back(MCFixup::create(0, Expr, isOpcode32(OpCode) ? FK_Data_4 : FK_Data_2, true));
    return 0;
  }

  // Constant immediate.
  int64_t Result;
  if (Expr->evaluateAsAbsolute(Result))
    return Result;

  return 0;
}

unsigned SuperHMCCodeEmitter::getMachineOpValue(const MCInst &MI, 
                             const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const {
  if (MO.isReg())
      return Ctx.getRegisterInfo()->getEncodingValue(MO.getReg());

  if (MO.isImm())
    return MO.getImm();

  assert(MO.isExpr() && "Expected Expression");
  return getExprOpValue(MI, MO.getExpr(), Fixups, STI);
}

MCCodeEmitter *llvm::createSuperHMCCodeEmitter(const MCInstrInfo &MCII,
                                              MCContext &Ctx) {
  return new SuperHMCCodeEmitter(MCII, Ctx);
}