//===- DPUOperandDecoder.cpp - Disassembler for DPU -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the DPU Disassembler.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "DPU-disassembler"

#include "DPUOperandDecoder.h"
#include "DPUDisassembler.h"
#include "MCTargetDesc/DPUMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_ENUM
#include "DPUGenInstrInfo.inc"

#define GET_REGINFO_ENUM
#include "DPUGenRegisterInfo.inc"

// Decoded registers, with respect to their indexing in the DPU. See
// DPURegisterInfo.td
static const unsigned GP32RegisterMap[] = {
    DPU::R0,  DPU::R1,  DPU::R2,  DPU::R3,   DPU::R4,  DPU::R5,   DPU::R6,
    DPU::R7,  DPU::R8,  DPU::R9,  DPU::R10,  DPU::R11, DPU::R12,  DPU::R13,
    DPU::R14, DPU::R15, DPU::R16, DPU::R17,  DPU::R18, DPU::R19,  DPU::R20,
    DPU::R21, DPU::R22, DPU::R23, DPU::ZERO, DPU::ONE, DPU::LNEG, DPU::MNEG,
    DPU::ID,  DPU::ID2, DPU::ID4, DPU::ID8};

// Decoded registers, with respect to their indexing in the DPU. See
// DPURegisterInfo.td
static const unsigned GP64RegisterMap[] = {
    DPU::D0,  DPU::D2,  DPU::D4,  DPU::D6,  DPU::D8,  DPU::D10,
    DPU::D12, DPU::D14, DPU::D16, DPU::D18, DPU::D20, DPU::D22};

static MCDisassembler::DecodeStatus Decode_32RegisterClass(MCInst &MI,
                                                           unsigned RegNo) {
  MI.addOperand(MCOperand::createReg(GP32RegisterMap[RegNo]));
  return MCDisassembler::Success;
}

static MCDisassembler::DecodeStatus Decode_OP_REG(MCInst &MI, unsigned RegNo) {
  return Decode_32RegisterClass(MI, RegNo);
}

static MCDisassembler::DecodeStatus Decode_GP_REG(MCInst &MI, unsigned RegNo) {
  if (RegNo >= 24) {
    LLVM_DEBUG(dbgs() << "ERROR: register #" << RegNo
                      << " is not a valid GP_REG\n");
    return MCDisassembler::Fail;
  }
  return Decode_32RegisterClass(MI, RegNo);
}

static MCDisassembler::DecodeStatus DecodeZeroOr_GP_REG(MCInst &MI,
                                                        unsigned RegNo) {
  if (RegNo > 24) {
    LLVM_DEBUG(dbgs() << "ERROR: register #" << RegNo
                      << " is not a valid GP_REG nor zero\n");
    return MCDisassembler::Fail;
  }
  return Decode_32RegisterClass(MI, RegNo);
}

static MCDisassembler::DecodeStatus Decode_GP64_REG(MCInst &MI,
                                                    unsigned RegNo) {
  // Register index must be in the allowed range of doubles and be even.
  if ((RegNo > 22) || ((RegNo & 1) == 1)) {
    LLVM_DEBUG(dbgs() << "ERROR: register #" << RegNo
                      << " is not a valid GP64_REG\n");
    return MCDisassembler::Fail;
  }
  MI.addOperand(MCOperand::createReg(GP64RegisterMap[RegNo >> 1]));
  return MCDisassembler::Success;
}

static bool tryAddingSymbolicOperand(int64_t Value, bool IsBranch,
                                     uint64_t Address, uint64_t Offset,
                                     uint64_t Width, MCInst &MI,
                                     const MCDisassembler *Decoder) {
  return Decoder->tryAddingSymbolicOperand(MI, Value, Address, IsBranch, Offset,
                                           Width);
}

static MCDisassembler::DecodeStatus DecodePC(MCInst &MI, unsigned Insn,
                                             uint64_t Address,
                                             const MCDisassembler *Decoder) {
  if (!tryAddingSymbolicOperand(Insn, true, Address, 2, 23, MI, Decoder))
    MI.addOperand(MCOperand::createImm(Insn));
  return MCDisassembler::Success;
}

MCDisassembler::DecodeStatus DPUOperandDecoder::Decode_endian(MCInst &MI,
                                                              uint64_t Value) {
  // 0 = little endian
  LLVM_DEBUG(dbgs() << "endian << " << Value << "\n");
  if (Value > 1) {
    LLVM_DEBUG(dbgs() << "ERROR: endian value #" << Value
                      << " is not boolean\n");
    return MCDisassembler::Fail;
  }
  MI.addOperand(MCOperand::createImm(Value));
  return MCDisassembler::Success;
}

MCDisassembler::DecodeStatus DPUOperandDecoder::Decode_ra(llvm::MCInst &MI,
                                                          uint64_t Value) {
  LLVM_DEBUG(dbgs() << "ra << " << Value << "\n");
  return Decode_OP_REG(MI, (unsigned)Value);
}

MCDisassembler::DecodeStatus DPUOperandDecoder::Decode_rb(llvm::MCInst &MI,
                                                          uint64_t Value) {
  LLVM_DEBUG(dbgs() << "rb << " << Value << "\n");
  return Decode_GP_REG(MI, (unsigned)Value);
}

MCDisassembler::DecodeStatus DPUOperandDecoder::Decode_db(llvm::MCInst &MI,
                                                          uint64_t Value) {
  LLVM_DEBUG(dbgs() << "db << " << Value << "\n");
  return Decode_GP64_REG(MI, (unsigned)Value);
}

MCDisassembler::DecodeStatus DPUOperandDecoder::Decode_imm(MCInst &MI,
                                                           int32_t Value) {
  LLVM_DEBUG(dbgs() << "imm << " << Value << "\n");
  MI.addOperand(MCOperand::createImm(Value));
  return MCDisassembler::Success;
}

MCDisassembler::DecodeStatus DPUOperandDecoder::Decode_immDma(MCInst &MI,
                                                              int32_t Value) {
  LLVM_DEBUG(dbgs() << "immDMA << " << Value << "\n");
  MI.addOperand(MCOperand::createImm(Value));
  return MCDisassembler::Success;
}

MCDisassembler::DecodeStatus
DPUOperandDecoder::Decode_imm(MCInst &MI, int32_t Value, int32_t ValueSize) {
  LLVM_DEBUG(dbgs() << "imm << " << Value << "\n");
  Value = Value | ((-(Value >> (ValueSize - 1))) << ValueSize);
  MI.addOperand(MCOperand::createImm(Value));
  return MCDisassembler::Success;
}

MCDisassembler::DecodeStatus
DPUOperandDecoder::Decode_cc(llvm::MCInst &MI, ConditionClass ccClass,
                             uint64_t Value) {
  LLVM_DEBUG(dbgs() << "cc << class=" << ccClass << " cc=" << Value << "\n");
  if (Value >= DPUAsmCondition::nrEncodingValue) {
    LLVM_DEBUG(dbgs() << "condition #" << Value
                      << " is not a valid condition value");
    return MCDisassembler::Fail;
  }
  int64_t encoding = DPUAsmCondition::getDecoding((Condition)Value, ccClass);
  MI.addOperand(MCOperand::createImm(encoding));
  return MCDisassembler::Success;
}

MCDisassembler::DecodeStatus DPUOperandDecoder::Decode_false_cc(MCInst &MI) {
  MI.addOperand(MCOperand::createImm(Condition::False));
  return MCDisassembler::Success;
}

#define FIXUP_PC(pc) (0x80000000 | (unsigned)((pc)*8))

MCDisassembler::DecodeStatus DPUOperandDecoder::Decode_pc(llvm::MCInst &MI,
                                                          uint64_t Value) {
  LLVM_DEBUG(dbgs() << "PC << " << Value << "\n");
  DecodePC(MI, FIXUP_PC(Value), Address, Decoder);
  return MCDisassembler::Success;
}

static MCDisassembler::DecodeStatus DecodeOff(MCInst &MI, int32_t Value,
                                              const MCDisassembler *Decoder) {
  switch (MI.getOpcode()) {
  case DPU::CALLrri:
  case DPU::JUMPri:
  case DPU::CALLzri: {
    const MCOperand &lastOperand = MI.getOperand(MI.getNumOperands() - 1);
    if (lastOperand.isReg() && lastOperand.getReg() == DPU::ZERO &&
        tryAddingSymbolicOperand(FIXUP_PC(Value), true, 0, 2, 23, MI, Decoder))
      return MCDisassembler::Success;
  }
  default:
    break;
  }
  MI.addOperand(MCOperand::createImm(Value));
  return MCDisassembler::Success;
}

MCDisassembler::DecodeStatus DPUOperandDecoder::Decode_off(llvm::MCInst &MI,
                                                           int32_t Value,
                                                           int32_t ValueSize) {
  LLVM_DEBUG(dbgs() << "off << " << Value << "\n");
  Value = Value | ((-(Value >> (ValueSize - 1))) << ValueSize);
  DecodeOff(MI, Value, Decoder);
  return MCDisassembler::Success;
}

MCDisassembler::DecodeStatus DPUOperandDecoder::Decode_off(llvm::MCInst &MI,
                                                           int32_t Value) {
  LLVM_DEBUG(dbgs() << "off << " << Value << "\n");
  DecodeOff(MI, Value, Decoder);
  return MCDisassembler::Success;
}

MCDisassembler::DecodeStatus DPUOperandDecoder::Decode_rc(llvm::MCInst &MI,
                                                          uint64_t Value) {
  LLVM_DEBUG(dbgs() << "rc << " << Value << "\n");
  return DecodeZeroOr_GP_REG(MI, (unsigned)Value);
}

MCDisassembler::DecodeStatus DPUOperandDecoder::Decode_dc(llvm::MCInst &MI,
                                                          uint64_t Value) {
  LLVM_DEBUG(dbgs() << "dc << " << Value << "\n");
  return Decode_GP64_REG(MI, (unsigned)Value);
}

MCDisassembler::DecodeStatus DPUOperandDecoder::Decode_shift(llvm::MCInst &MI,
                                                             uint64_t Value) {
  LLVM_DEBUG(dbgs() << "shift << " << Value << "\n");
  MI.addOperand(MCOperand::createImm(Value));
  return MCDisassembler::Success;
}

MCDisassembler::DecodeStatus DPUOperandDecoder::Decode_zero(llvm::MCInst &MI,
                                                            uint64_t Value) {
  LLVM_DEBUG(dbgs() << "zero << " << Value << "\n");
  MI.addOperand(MCOperand::createReg(GP32RegisterMap[Value]));
  return MCDisassembler::Success;
}
