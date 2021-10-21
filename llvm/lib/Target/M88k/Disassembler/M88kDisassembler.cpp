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

static const uint16_t XRDecoderTable[] = {
    M88k::X0,  M88k::X1,  M88k::X2,  M88k::X3,  M88k::X4,  M88k::X5,
    M88k::X6,  M88k::X7,  M88k::X8,  M88k::X9,  M88k::X10, M88k::X11,
    M88k::X12, M88k::X13, M88k::X14, M88k::X15,

    M88k::X16, M88k::X17, M88k::X18, M88k::X19, M88k::X20, M88k::X21,
    M88k::X22, M88k::X23, M88k::X24, M88k::X25, M88k::X26, M88k::X27,
    M88k::X28, M88k::X29, M88k::X30, M88k::X31,
};

static const uint16_t CRDecoderTable[] = {
    M88k::CR0,  M88k::CR1,  M88k::CR2,  M88k::CR3,  M88k::CR4,  M88k::CR5,
    M88k::CR6,  M88k::CR7,  M88k::CR8,  M88k::CR9,  M88k::CR10, M88k::CR11,
    M88k::CR12, M88k::CR13, M88k::CR14, M88k::CR15,

    M88k::CR16, M88k::CR17, M88k::CR18, M88k::CR19, M88k::CR20, M88k::CR21,
    M88k::CR22, M88k::CR23, M88k::CR24, M88k::CR25, M88k::CR26, M88k::CR27,
    M88k::CR28, M88k::CR29, M88k::CR30, M88k::CR31,

    M88k::CR32, M88k::CR33, M88k::CR34, M88k::CR35, M88k::CR36, M88k::CR37,
    M88k::CR38, M88k::CR39,

    M88k::CR40, M88k::CR41, M88k::CR42, M88k::CR43, M88k::CR44, M88k::CR45,
    M88k::CR46, M88k::CR47, M88k::CR48, M88k::CR49, M88k::CR50, M88k::CR51,
    M88k::CR52, M88k::CR53, M88k::CR54, M88k::CR55, M88k::CR56, M88k::CR57,
    M88k::CR58, M88k::CR59, M88k::CR60, M88k::CR61, M88k::CR62, M88k::CR63};

static const uint16_t FCRDecoderTable[] = {
    M88k::FCR0,  M88k::FCR1,  M88k::FCR2,  M88k::FCR3,  M88k::FCR4,
    M88k::FCR5,  M88k::FCR6,  M88k::FCR7,  M88k::FCR8,  M88k::FCR9,
    M88k::FCR10, M88k::FCR11, M88k::FCR12, M88k::FCR13, M88k::FCR14,
    M88k::FCR15,

    M88k::FCR16, M88k::FCR17, M88k::FCR18, M88k::FCR19, M88k::FCR20,
    M88k::FCR21, M88k::FCR22, M88k::FCR23, M88k::FCR24, M88k::FCR25,
    M88k::FCR26, M88k::FCR27, M88k::FCR28, M88k::FCR29, M88k::FCR30,
    M88k::FCR31,

    M88k::FCR32, M88k::FCR33, M88k::FCR34, M88k::FCR35, M88k::FCR36,
    M88k::FCR37, M88k::FCR38, M88k::FCR39,

    M88k::FCR40, M88k::FCR41, M88k::FCR42, M88k::FCR43, M88k::FCR44,
    M88k::FCR45, M88k::FCR46, M88k::FCR47, M88k::FCR48, M88k::FCR49,
    M88k::FCR50, M88k::FCR51, M88k::FCR52, M88k::FCR53, M88k::FCR54,
    M88k::FCR55, M88k::FCR56, M88k::FCR57, M88k::FCR58, M88k::FCR59,
    M88k::FCR60, M88k::FCR61, M88k::FCR62, M88k::FCR63};

static DecodeStatus decodeGPRRegisterClass(MCInst &Inst, uint64_t RegNo,
                                           uint64_t Address,
                                           const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  unsigned Register = GPRDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus decodeXRRegisterClass(MCInst &Inst, uint64_t RegNo,
                                          uint64_t Address,
                                          const void *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  unsigned Register = XRDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus decodeCRRegisterClass(MCInst &Inst, uint64_t RegNo,
                                          uint64_t Address,
                                          const void *Decoder) {
  if (RegNo > 63)
    return MCDisassembler::Fail;

  unsigned Register = CRDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus decodeFCRRegisterClass(MCInst &Inst, uint64_t RegNo,
                                           uint64_t Address,
                                           const void *Decoder) {
  if (RegNo > 63)
    return MCDisassembler::Fail;

  unsigned Register = FCRDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Register));
  return MCDisassembler::Success;
}

// TODO More checks.
static DecodeStatus decodeGPR64RegisterClass(MCInst &Inst, uint64_t RegNo,
                                             uint64_t Address,
                                             const void *Decoder) {
  return decodeGPRRegisterClass(Inst, RegNo, Address, Decoder);
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
