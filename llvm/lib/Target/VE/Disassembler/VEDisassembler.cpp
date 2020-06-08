//===- VEDisassembler.cpp - Disassembler for VE -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is part of the VE Disassembler.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/VEMCTargetDesc.h"
#include "TargetInfo/VETargetInfo.h"
#include "VE.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "ve-disassembler"

typedef MCDisassembler::DecodeStatus DecodeStatus;

namespace {

/// A disassembler class for VE.
class VEDisassembler : public MCDisassembler {
public:
  VEDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx)
      : MCDisassembler(STI, Ctx) {}
  virtual ~VEDisassembler() {}

  DecodeStatus getInstruction(MCInst &Instr, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &CStream) const override;
};
} // namespace

static MCDisassembler *createVEDisassembler(const Target &T,
                                            const MCSubtargetInfo &STI,
                                            MCContext &Ctx) {
  return new VEDisassembler(STI, Ctx);
}

extern "C" void LLVMInitializeVEDisassembler() {
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(getTheVETarget(),
                                         createVEDisassembler);
}

static const unsigned I32RegDecoderTable[] = {
    VE::SW0,  VE::SW1,  VE::SW2,  VE::SW3,  VE::SW4,  VE::SW5,  VE::SW6,
    VE::SW7,  VE::SW8,  VE::SW9,  VE::SW10, VE::SW11, VE::SW12, VE::SW13,
    VE::SW14, VE::SW15, VE::SW16, VE::SW17, VE::SW18, VE::SW19, VE::SW20,
    VE::SW21, VE::SW22, VE::SW23, VE::SW24, VE::SW25, VE::SW26, VE::SW27,
    VE::SW28, VE::SW29, VE::SW30, VE::SW31, VE::SW32, VE::SW33, VE::SW34,
    VE::SW35, VE::SW36, VE::SW37, VE::SW38, VE::SW39, VE::SW40, VE::SW41,
    VE::SW42, VE::SW43, VE::SW44, VE::SW45, VE::SW46, VE::SW47, VE::SW48,
    VE::SW49, VE::SW50, VE::SW51, VE::SW52, VE::SW53, VE::SW54, VE::SW55,
    VE::SW56, VE::SW57, VE::SW58, VE::SW59, VE::SW60, VE::SW61, VE::SW62,
    VE::SW63};

static const unsigned I64RegDecoderTable[] = {
    VE::SX0,  VE::SX1,  VE::SX2,  VE::SX3,  VE::SX4,  VE::SX5,  VE::SX6,
    VE::SX7,  VE::SX8,  VE::SX9,  VE::SX10, VE::SX11, VE::SX12, VE::SX13,
    VE::SX14, VE::SX15, VE::SX16, VE::SX17, VE::SX18, VE::SX19, VE::SX20,
    VE::SX21, VE::SX22, VE::SX23, VE::SX24, VE::SX25, VE::SX26, VE::SX27,
    VE::SX28, VE::SX29, VE::SX30, VE::SX31, VE::SX32, VE::SX33, VE::SX34,
    VE::SX35, VE::SX36, VE::SX37, VE::SX38, VE::SX39, VE::SX40, VE::SX41,
    VE::SX42, VE::SX43, VE::SX44, VE::SX45, VE::SX46, VE::SX47, VE::SX48,
    VE::SX49, VE::SX50, VE::SX51, VE::SX52, VE::SX53, VE::SX54, VE::SX55,
    VE::SX56, VE::SX57, VE::SX58, VE::SX59, VE::SX60, VE::SX61, VE::SX62,
    VE::SX63};

static const unsigned F32RegDecoderTable[] = {
    VE::SF0,  VE::SF1,  VE::SF2,  VE::SF3,  VE::SF4,  VE::SF5,  VE::SF6,
    VE::SF7,  VE::SF8,  VE::SF9,  VE::SF10, VE::SF11, VE::SF12, VE::SF13,
    VE::SF14, VE::SF15, VE::SF16, VE::SF17, VE::SF18, VE::SF19, VE::SF20,
    VE::SF21, VE::SF22, VE::SF23, VE::SF24, VE::SF25, VE::SF26, VE::SF27,
    VE::SF28, VE::SF29, VE::SF30, VE::SF31, VE::SF32, VE::SF33, VE::SF34,
    VE::SF35, VE::SF36, VE::SF37, VE::SF38, VE::SF39, VE::SF40, VE::SF41,
    VE::SF42, VE::SF43, VE::SF44, VE::SF45, VE::SF46, VE::SF47, VE::SF48,
    VE::SF49, VE::SF50, VE::SF51, VE::SF52, VE::SF53, VE::SF54, VE::SF55,
    VE::SF56, VE::SF57, VE::SF58, VE::SF59, VE::SF60, VE::SF61, VE::SF62,
    VE::SF63};

static DecodeStatus DecodeI32RegisterClass(MCInst &Inst, unsigned RegNo,
                                           uint64_t Address,
                                           const void *Decoder) {
  if (RegNo > 63)
    return MCDisassembler::Fail;
  unsigned Reg = I32RegDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeI64RegisterClass(MCInst &Inst, unsigned RegNo,
                                           uint64_t Address,
                                           const void *Decoder) {
  if (RegNo > 63)
    return MCDisassembler::Fail;
  unsigned Reg = I64RegDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeF32RegisterClass(MCInst &Inst, unsigned RegNo,
                                           uint64_t Address,
                                           const void *Decoder) {
  if (RegNo > 63)
    return MCDisassembler::Fail;
  unsigned Reg = F32RegDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeLoadI32(MCInst &Inst, uint64_t insn, uint64_t Address,
                                  const void *Decoder);
static DecodeStatus DecodeStoreI32(MCInst &Inst, uint64_t insn,
                                   uint64_t Address, const void *Decoder);
static DecodeStatus DecodeLoadI64(MCInst &Inst, uint64_t insn, uint64_t Address,
                                  const void *Decoder);
static DecodeStatus DecodeStoreI64(MCInst &Inst, uint64_t insn,
                                   uint64_t Address, const void *Decoder);
static DecodeStatus DecodeLoadF32(MCInst &Inst, uint64_t insn, uint64_t Address,
                                  const void *Decoder);
static DecodeStatus DecodeStoreF32(MCInst &Inst, uint64_t insn,
                                   uint64_t Address, const void *Decoder);
static DecodeStatus DecodeSIMM7(MCInst &Inst, uint64_t insn, uint64_t Address,
                                const void *Decoder);

#include "VEGenDisassemblerTables.inc"

/// Read four bytes from the ArrayRef and return 32 bit word.
static DecodeStatus readInstruction64(ArrayRef<uint8_t> Bytes, uint64_t Address,
                                      uint64_t &Size, uint64_t &Insn,
                                      bool IsLittleEndian) {
  // We want to read exactly 8 Bytes of data.
  if (Bytes.size() < 8) {
    Size = 0;
    return MCDisassembler::Fail;
  }

  Insn = IsLittleEndian
             ? ((uint64_t)Bytes[0] << 0) | ((uint64_t)Bytes[1] << 8) |
                   ((uint64_t)Bytes[2] << 16) | ((uint64_t)Bytes[3] << 24) |
                   ((uint64_t)Bytes[4] << 32) | ((uint64_t)Bytes[5] << 40) |
                   ((uint64_t)Bytes[6] << 48) | ((uint64_t)Bytes[7] << 56)
             : ((uint64_t)Bytes[7] << 0) | ((uint64_t)Bytes[6] << 8) |
                   ((uint64_t)Bytes[5] << 16) | ((uint64_t)Bytes[4] << 24) |
                   ((uint64_t)Bytes[3] << 32) | ((uint64_t)Bytes[2] << 40) |
                   ((uint64_t)Bytes[1] << 48) | ((uint64_t)Bytes[0] << 56);

  return MCDisassembler::Success;
}

DecodeStatus VEDisassembler::getInstruction(MCInst &Instr, uint64_t &Size,
                                            ArrayRef<uint8_t> Bytes,
                                            uint64_t Address,
                                            raw_ostream &CStream) const {
  uint64_t Insn;
  bool isLittleEndian = getContext().getAsmInfo()->isLittleEndian();
  DecodeStatus Result =
      readInstruction64(Bytes, Address, Size, Insn, isLittleEndian);
  if (Result == MCDisassembler::Fail)
    return MCDisassembler::Fail;

  // Calling the auto-generated decoder function.

  Result = decodeInstruction(DecoderTableVE64, Instr, Insn, Address, this, STI);

  if (Result != MCDisassembler::Fail) {
    Size = 8;
    return Result;
  }

  return MCDisassembler::Fail;
}

typedef DecodeStatus (*DecodeFunc)(MCInst &MI, unsigned RegNo, uint64_t Address,
                                   const void *Decoder);

static DecodeStatus DecodeASX(MCInst &MI, uint64_t insn, uint64_t Address,
                              const void *Decoder) {
  unsigned sy = fieldFromInstruction(insn, 40, 7);
  bool cy = fieldFromInstruction(insn, 47, 1);
  unsigned sz = fieldFromInstruction(insn, 32, 7);
  bool cz = fieldFromInstruction(insn, 39, 1);
  uint64_t simm32 = SignExtend64<32>(fieldFromInstruction(insn, 0, 32));
  DecodeStatus status;

  // Decode sz.
  if (cz) {
    status = DecodeI64RegisterClass(MI, sz, Address, Decoder);
    if (status != MCDisassembler::Success)
      return status;
  } else {
    MI.addOperand(MCOperand::createImm(0));
  }

  // Decode sy.
  if (cy) {
    status = DecodeI64RegisterClass(MI, sy, Address, Decoder);
    if (status != MCDisassembler::Success)
      return status;
  } else {
    MI.addOperand(MCOperand::createImm(SignExtend32<7>(sy)));
  }

  // Decode simm32.
  MI.addOperand(MCOperand::createImm(simm32));

  return MCDisassembler::Success;
}

static DecodeStatus DecodeMem(MCInst &MI, uint64_t insn, uint64_t Address,
                              const void *Decoder, bool isLoad,
                              DecodeFunc DecodeSX) {
  unsigned sx = fieldFromInstruction(insn, 48, 7);

  DecodeStatus status;
  if (isLoad) {
    status = DecodeSX(MI, sx, Address, Decoder);
    if (status != MCDisassembler::Success)
      return status;
  }

  status = DecodeASX(MI, insn, Address, Decoder);
  if (status != MCDisassembler::Success)
    return status;

  if (!isLoad) {
    status = DecodeSX(MI, sx, Address, Decoder);
    if (status != MCDisassembler::Success)
      return status;
  }
  return MCDisassembler::Success;
}

static DecodeStatus DecodeLoadI32(MCInst &Inst, uint64_t insn, uint64_t Address,
                                  const void *Decoder) {
  return DecodeMem(Inst, insn, Address, Decoder, true, DecodeI32RegisterClass);
}

static DecodeStatus DecodeStoreI32(MCInst &Inst, uint64_t insn,
                                   uint64_t Address, const void *Decoder) {
  return DecodeMem(Inst, insn, Address, Decoder, false, DecodeI32RegisterClass);
}

static DecodeStatus DecodeLoadI64(MCInst &Inst, uint64_t insn, uint64_t Address,
                                  const void *Decoder) {
  return DecodeMem(Inst, insn, Address, Decoder, true, DecodeI64RegisterClass);
}

static DecodeStatus DecodeStoreI64(MCInst &Inst, uint64_t insn,
                                   uint64_t Address, const void *Decoder) {
  return DecodeMem(Inst, insn, Address, Decoder, false, DecodeI64RegisterClass);
}

static DecodeStatus DecodeLoadF32(MCInst &Inst, uint64_t insn, uint64_t Address,
                                  const void *Decoder) {
  return DecodeMem(Inst, insn, Address, Decoder, true, DecodeF32RegisterClass);
}

static DecodeStatus DecodeStoreF32(MCInst &Inst, uint64_t insn,
                                   uint64_t Address, const void *Decoder) {
  return DecodeMem(Inst, insn, Address, Decoder, false, DecodeF32RegisterClass);
}

static DecodeStatus DecodeSIMM7(MCInst &MI, uint64_t insn, uint64_t Address,
                                const void *Decoder) {
  uint64_t tgt = SignExtend64<7>(insn);
  MI.addOperand(MCOperand::createImm(tgt));
  return MCDisassembler::Success;
}
