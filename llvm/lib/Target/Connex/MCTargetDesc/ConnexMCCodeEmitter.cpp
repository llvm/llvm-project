//===-- ConnexMCCodeEmitter.cpp - Convert Connex code to machine code -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ConnexMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ConnexMCFixups.h"
#include "MCTargetDesc/ConnexMCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "mccodeemitter"

namespace {

class ConnexMCCodeEmitter : public MCCodeEmitter {
  const MCInstrInfo &MCII;
  const MCRegisterInfo &MRI;
  bool IsLittleEndian;
  MCContext &Ctx; // Inspired from MCTargetDesc/BPFMCCodeEmitter.cpp (Oct 2025)

public:
  ConnexMCCodeEmitter(const MCInstrInfo &mcii, const MCRegisterInfo &mri,
                      bool IsLittleEndian, MCContext &ctx)
      : MCII(mcii), MRI(mri), IsLittleEndian(IsLittleEndian), Ctx(ctx) {}

  ConnexMCCodeEmitter(const ConnexMCCodeEmitter &) = delete;

  void operator=(const ConnexMCCodeEmitter &) = delete;

  ~ConnexMCCodeEmitter() override = default;

  // getBinaryCodeForInstr - TableGen'erated function for getting the
  // binary encoding for an instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;

  // getMachineOpValue - Return binary encoding of operand. If the machin
  // operand requires relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;

  uint64_t getMemoryOpValue(const MCInst &MI, unsigned Op,
                            SmallVectorImpl<MCFixup> &Fixups,
                            const MCSubtargetInfo &STI) const;

  void encodeInstruction(const MCInst &Inst, SmallVectorImpl< char > &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;
};

} // end anonymous namespace

MCCodeEmitter *llvm::createConnexMCCodeEmitter(const MCInstrInfo &MCII,
                                               MCContext &Ctx) {
  return new ConnexMCCodeEmitter(MCII, *(Ctx.getRegisterInfo()), true, Ctx);
}

static void addFixup(SmallVectorImpl<MCFixup> &Fixups, uint32_t Offset,
                     const MCExpr *Value, uint16_t Kind, bool PCRel = false) {
  Fixups.push_back(MCFixup::create(Offset, Value, Kind, PCRel));
}

unsigned ConnexMCCodeEmitter::getMachineOpValue(const MCInst &MI,
                                             const MCOperand &MO,
                                             SmallVectorImpl<MCFixup> &Fixups,
                                             const MCSubtargetInfo &STI) const {
  if (MO.isReg())
    return MRI.getEncodingValue(MO.getReg());
  if (MO.isImm()) {
    uint64_t Imm = MO.getImm();
    uint64_t High32Bits = Imm >> 32, High33Bits = Imm >> 31;
    if (MI.getOpcode() != Connex::LD_imm64 && High32Bits != 0 &&
        High33Bits != 0x1FFFFFFFFULL) {
      Ctx.reportWarning(MI.getLoc(),
                        "immediate out of range, shall fit in 32 bits");
    }
    return static_cast<unsigned>(Imm);
  }

  assert(MO.isExpr());

  const MCExpr *Expr = MO.getExpr();

  assert(Expr->getKind() == MCExpr::SymbolRef);

  if (MI.getOpcode() == Connex::JAL)
    // func call name
    addFixup(Fixups, 0, Expr, FK_Data_4, true);
  else if (MI.getOpcode() == Connex::LD_imm64)
    addFixup(Fixups, 0, Expr, FK_SecRel_8);
  /*
  else if (MI.getOpcode() == Connex::JMPL)
    addFixup(Fixups, 0, Expr, Connex::FK_Connex_PCRel_4, true);
  */
  else
    // bb label
    addFixup(Fixups, 0, Expr, FK_Data_2, true);

  return 0;
}

static uint8_t SwapBits(uint8_t Val) {
  return (Val & 0x0F) << 4 | (Val & 0xF0) >> 4;
}

void ConnexMCCodeEmitter::encodeInstruction(const MCInst &MI, SmallVectorImpl< char > &CB,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
  unsigned Opcode = MI.getOpcode();
  raw_svector_ostream OS(CB); // Inspired from MCTargetDesc/BPFMCCodeEmitter.cpp
  support::endian::Writer OSE(OS,
                              IsLittleEndian ? llvm::endianness::little
                                             : llvm::endianness::big);

  if (Opcode == Connex::LD_imm64 || Opcode == Connex::LD_pseudo) {
    uint64_t Value = getBinaryCodeForInstr(MI, Fixups, STI);
    OS << char(Value >> 56);
    if (IsLittleEndian)
      OS << char((Value >> 48) & 0xff);
    else
      OS << char(SwapBits((Value >> 48) & 0xff));
    OSE.write<uint16_t>(0);
    OSE.write<uint32_t>(Value & 0xffffFFFF);

    const MCOperand &MO = MI.getOperand(1);
    uint64_t Imm = MO.isImm() ? MO.getImm() : 0;
    OSE.write<uint8_t>(0);
    OSE.write<uint8_t>(0);
    OSE.write<uint16_t>(0);
    OSE.write<uint32_t>(Imm >> 32);
  } else {
    // Get instruction encoding and emit it
    uint64_t Value = getBinaryCodeForInstr(MI, Fixups, STI);
    OS << char(Value >> 56);
    if (IsLittleEndian)
      OS << char((Value >> 48) & 0xff);
    else
      OS << char(SwapBits((Value >> 48) & 0xff));
    OSE.write<uint16_t>((Value >> 32) & 0xffff);
    OSE.write<uint32_t>(Value & 0xffffFFFF);
  }
}

// Encode Connex Memory Operand
uint64_t
ConnexMCCodeEmitter::getMemoryOpValue(const MCInst &MI, unsigned Op,
                                      SmallVectorImpl<MCFixup> &Fixups,
                                      const MCSubtargetInfo &STI) const {
  uint64_t Encoding;
  const MCOperand Op1 = MI.getOperand(1);
  assert(Op1.isReg() && "First operand is not register.");
  Encoding = MRI.getEncodingValue(Op1.getReg());
  Encoding <<= 16;
  MCOperand Op2 = MI.getOperand(2);
  assert(Op2.isImm() && "Second operand is not immediate.");
  Encoding |= Op2.getImm() & 0xffff;
  return Encoding;
}

#include "ConnexGenMCCodeEmitter.inc"
