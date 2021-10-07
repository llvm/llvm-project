//===-- M88kDisassembler.cpp - Disassembler for M88k ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "M88k.h"
#include "MCTargetDesc/M88kMCTargetDesc.h"
#include "TargetInfo/M88kTargetInfo.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "m88k-disassembler"

using DecodeStatus = MCDisassembler::DecodeStatus;

namespace {

class M88kDisassembler : public MCDisassembler {
public:
  M88kDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx)
      : MCDisassembler(STI, Ctx) {}
  ~M88kDisassembler() override = default;

  DecodeStatus getInstruction(MCInst &instr, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &CStream) const override;
};

} // end anonymous namespace

static MCDisassembler *createM88kDisassembler(const Target &T,
                                              const MCSubtargetInfo &STI,
                                              MCContext &Ctx) {
  return new M88kDisassembler(STI, Ctx);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeM88kDisassembler() {
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(getTheM88kTarget(),
                                         createM88kDisassembler);
}

static const uint16_t GPRDecoderTable[] = {
    M88k::R0,  M88k::R1,  M88k::R2,  M88k::R3,  M88k::R4,  M88k::R5,
    M88k::R6,  M88k::R7,  M88k::R8,  M88k::R9,  M88k::R10, M88k::R11,
    M88k::R12, M88k::R13, M88k::R14, M88k::R15,

    M88k::R16, M88k::R17, M88k::R18, M88k::R19, M88k::R20, M88k::R21,
    M88k::R22, M88k::R23, M88k::R24, M88k::R25, M88k::R26, M88k::R27,
    M88k::R28, M88k::R29, M88k::R30, M88k::R31,
};

static DecodeStatus DecodeGPRRegisterClass(MCInst &Inst, uint64_t RegNo,
                                           uint64_t Address,
                                           const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  unsigned Register = GPRDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Register));
  return MCDisassembler::Success;
}

// TODO More checks.
static DecodeStatus DecodeGPR64RegisterClass(MCInst &Inst, uint64_t RegNo,
                                             uint64_t Address,
                                             const void *Decoder) {
  return DecodeGPRRegisterClass(Inst, RegNo, Address, Decoder);
}

static DecodeStatus DecodeGPRF32RegisterClass(MCInst &Inst, uint64_t RegNo,
                                              uint64_t Address,
                                              const void *Decoder) {
  return DecodeGPRRegisterClass(Inst, RegNo, Address, Decoder);
}

static DecodeStatus DecodeGPRF64RegisterClass(MCInst &Inst, uint64_t RegNo,
                                              uint64_t Address,
                                              const void *Decoder) {
  return DecodeGPRRegisterClass(Inst, RegNo, Address, Decoder);
}

template <unsigned N>
static DecodeStatus decodeUImmOperand(MCInst &Inst, uint64_t Imm) {
  if (!isUInt<N>(Imm))
    return MCDisassembler::Fail;
  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeU5ImmOOperand(MCInst &Inst, uint64_t Imm,
                                        uint64_t Address, const void *Decoder) {
  return decodeUImmOperand<5>(Inst, Imm);
}

static DecodeStatus decodeU5ImmOperand(MCInst &Inst, uint64_t Imm,
                                       uint64_t Address, const void *Decoder) {
  return decodeUImmOperand<5>(Inst, Imm);
}

static DecodeStatus decodeU10ImmWOOperand(MCInst &Inst, uint64_t Imm,
                                          uint64_t Address,
                                          const void *Decoder) {
  return decodeUImmOperand<10>(Inst, Imm);
}

static DecodeStatus decodeU16ImmOperand(MCInst &Inst, uint64_t Imm,
                                        uint64_t Address, const void *Decoder) {
  return decodeUImmOperand<16>(Inst, Imm);
}

static DecodeStatus decodePC26BranchOperand(MCInst &Inst, uint64_t Imm,
                                            uint64_t Address,
                                            const void *Decoder) {
  if (!isUInt<26>(Imm))
    return MCDisassembler::Fail;
  Inst.addOperand(MCOperand::createImm(SignExtend64<26>(Imm)));
  return MCDisassembler::Success;
}

static DecodeStatus decodePC16BranchOperand(MCInst &Inst, uint64_t Imm,
                                            uint64_t Address,
                                            const void *Decoder) {
  if (!isUInt<16>(Imm))
    return MCDisassembler::Fail;
  Inst.addOperand(MCOperand::createImm(SignExtend64<16>(Imm)));
  return MCDisassembler::Success;
}

#include "M88kGenDisassemblerTables.inc"

DecodeStatus M88kDisassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                              ArrayRef<uint8_t> Bytes,
                                              uint64_t Address,
                                              raw_ostream &CS) const {
  // Instruction size is always 32 bit.
  if (Bytes.size() < 4) {
    Size = 0;
    return MCDisassembler::Fail;
  }
  Size = 4;

  // Construct the instruction.
  uint32_t Inst = 0;
  for (uint32_t I = 0; I < Size; ++I)
    Inst = (Inst << 8) | Bytes[I];

  return decodeInstruction(DecoderTableM88k32, MI, Inst, Address, this, STI);
}
