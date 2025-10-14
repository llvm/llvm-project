//===- AVRDisassembler.cpp - Disassembler for AVR ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is part of the AVR Disassembler.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/AVRMCTargetDesc.h"
#include "TargetInfo/AVRTargetInfo.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDecoder.h"
#include "llvm/MC/MCDecoderOps.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/TargetRegistry.h"

#include "llvm/Support/Compiler.h"

using namespace llvm;
using namespace llvm::MCD;

#define DEBUG_TYPE "avr-disassembler"

typedef MCDisassembler::DecodeStatus DecodeStatus;

namespace {

/// A disassembler class for AVR.
class AVRDisassembler : public MCDisassembler {
public:
  AVRDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx)
      : MCDisassembler(STI, Ctx) {}
  virtual ~AVRDisassembler() = default;

  DecodeStatus getInstruction(MCInst &Instr, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &CStream) const override;
};
} // namespace

static MCDisassembler *createAVRDisassembler(const Target &T,
                                             const MCSubtargetInfo &STI,
                                             MCContext &Ctx) {
  return new AVRDisassembler(STI, Ctx);
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeAVRDisassembler() {
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(getTheAVRTarget(),
                                         createAVRDisassembler);
}

static constexpr MCRegister GPRDecoderTable[] = {
    AVR::R0,  AVR::R1,  AVR::R2,  AVR::R3,  AVR::R4,  AVR::R5,  AVR::R6,
    AVR::R7,  AVR::R8,  AVR::R9,  AVR::R10, AVR::R11, AVR::R12, AVR::R13,
    AVR::R14, AVR::R15, AVR::R16, AVR::R17, AVR::R18, AVR::R19, AVR::R20,
    AVR::R21, AVR::R22, AVR::R23, AVR::R24, AVR::R25, AVR::R26, AVR::R27,
    AVR::R28, AVR::R29, AVR::R30, AVR::R31,
};

static constexpr MCRegister GPRPairDecoderTable[] = {
    AVR::R1R0,   AVR::R3R2,   AVR::R5R4,   AVR::R7R6,
    AVR::R9R8,   AVR::R11R10, AVR::R13R12, AVR::R15R14,
    AVR::R17R16, AVR::R19R18, AVR::R21R20, AVR::R23R22,
    AVR::R25R24, AVR::R27R26, AVR::R29R28, AVR::R31R30,
};

static DecodeStatus DecodeGPR8RegisterClass(MCInst &Inst, unsigned RegNo,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;

  unsigned Register = GPRDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Register));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeLD8RegisterClass(MCInst &Inst, unsigned RegNo,
                                           uint64_t Address,
                                           const MCDisassembler *Decoder) {
  assert(isUInt<4>(RegNo));
  // Only r16...r31 are legal.
  Inst.addOperand(MCOperand::createReg(GPRDecoderTable[16 + RegNo]));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeLD8loRegisterClass(MCInst &Inst, unsigned RegNo,
                                             uint64_t Address,
                                             const MCDisassembler *Decoder) {
  assert(isUInt<3>(RegNo));
  // Only r16...r23 are legal.
  Inst.addOperand(MCOperand::createReg(GPRDecoderTable[16 + RegNo]));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeDREGSRegisterClass(MCInst &Inst, unsigned RegNo,
                                             uint64_t Address,
                                             const MCDisassembler *Decoder) {
  assert(isUInt<4>(RegNo));
  Inst.addOperand(MCOperand::createReg(GPRPairDecoderTable[RegNo]));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeIWREGSRegisterClass(MCInst &Inst, unsigned RegNo,
                                              uint64_t Address,
                                              const MCDisassembler *Decoder) {
  assert(isUInt<2>(RegNo));
  // Only AVR::R25R24, AVR::R27R26, AVR::R29R28, AVR::R31R30 are legal.
  Inst.addOperand(MCOperand::createReg(GPRPairDecoderTable[12 + RegNo]));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeZREGRegisterClass(MCInst &Inst,
                                            const MCDisassembler *Decoder) {
  Inst.addOperand(MCOperand::createReg(AVR::R31R30));
  return MCDisassembler::Success;
}

static DecodeStatus decodeCallTarget(MCInst &Inst, unsigned Field,
                                     uint64_t Address,
                                     const MCDisassembler *Decoder) {
  // Call targets need to be shifted left by one so this needs a custom
  // decoder.
  Inst.addOperand(MCOperand::createImm(Field << 1));
  return MCDisassembler::Success;
}

static DecodeStatus decodeRelCondBrTarget7(MCInst &Inst, unsigned Field,
                                           uint64_t Address,
                                           const MCDisassembler *Decoder) {
  // The legal range is [-128, 126] (in bytes).
  Inst.addOperand(MCOperand::createImm(SignExtend32(Field, 7) * 2));
  return MCDisassembler::Success;
}

static DecodeStatus decodeRelCondBrTarget13(MCInst &Inst, unsigned Field,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  // The legal range is [-4096, 4094] (in bytes).
  Inst.addOperand(MCOperand::createImm(SignExtend32(Field, 12) * 2));
  return MCDisassembler::Success;
}

static DecodeStatus decodeMemri(MCInst &Inst, unsigned Insn, uint64_t Address,
                                const MCDisassembler *Decoder) {
  // As in the EncoderMethod `AVRMCCodeEmitter::encodeMemri`, the memory
  // address is encoded into 7-bit, in which bits 0-5 are the immediate offset,
  // and the bit-6 is the pointer register bit (Z=0, Y=1).
  if (Insn > 127)
    return MCDisassembler::Fail;

  // Append the base register operand.
  Inst.addOperand(
      MCOperand::createReg((Insn & 0x40) ? AVR::R29R28 : AVR::R31R30));
  // Append the immediate offset operand.
  Inst.addOperand(MCOperand::createImm(Insn & 0x3f));

  return MCDisassembler::Success;
}

static DecodeStatus decodeLoadStore(MCInst &Inst, unsigned Insn,
                                    uint64_t Address,
                                    const MCDisassembler *Decoder) {
  // Get the register will be loaded or stored.
  unsigned RegVal = GPRDecoderTable[(Insn >> 4) & 0x1f];

  // Decode LDD/STD with offset less than 8.
  if ((Insn & 0xf000) == 0x8000) {
    unsigned RegBase = (Insn & 0x8) ? AVR::R29R28 : AVR::R31R30;
    unsigned Offset = Insn & 7; // We need not consider offset > 7.
    if ((Insn & 0x200) == 0) {  // Decode LDD.
      Inst.setOpcode(AVR::LDDRdPtrQ);
      Inst.addOperand(MCOperand::createReg(RegVal));
      Inst.addOperand(MCOperand::createReg(RegBase));
      Inst.addOperand(MCOperand::createImm(Offset));
    } else { // Decode STD.
      Inst.setOpcode(AVR::STDPtrQRr);
      Inst.addOperand(MCOperand::createReg(RegBase));
      Inst.addOperand(MCOperand::createImm(Offset));
      Inst.addOperand(MCOperand::createReg(RegVal));
    }
    return MCDisassembler::Success;
  }

  // Decode the following 14 instructions. Bit 9 indicates load(0) or store(1),
  // bits 8~4 indicate the value register, bits 3-2 indicate the base address
  // register (11-X, 10-Y, 00-Z), bits 1~0 indicate the mode (00-basic,
  // 01-postinc, 10-predec).
  // ST X,  Rr : 1001 001r rrrr 1100
  // ST X+, Rr : 1001 001r rrrr 1101
  // ST -X, Rr : 1001 001r rrrr 1110
  // ST Y+, Rr : 1001 001r rrrr 1001
  // ST -Y, Rr : 1001 001r rrrr 1010
  // ST Z+, Rr : 1001 001r rrrr 0001
  // ST -Z, Rr : 1001 001r rrrr 0010
  // LD Rd, X  : 1001 000d dddd 1100
  // LD Rd, X+ : 1001 000d dddd 1101
  // LD Rd, -X : 1001 000d dddd 1110
  // LD Rd, Y+ : 1001 000d dddd 1001
  // LD Rd, -Y : 1001 000d dddd 1010
  // LD Rd, Z+ : 1001 000d dddd 0001
  // LD Rd, -Z : 1001 000d dddd 0010
  if ((Insn & 0xfc00) != 0x9000 || (Insn & 0xf) == 0)
    return MCDisassembler::Fail;

  // Get the base address register.
  unsigned RegBase;
  switch (Insn & 0xc) {
  case 0xc:
    RegBase = AVR::R27R26;
    break;
  case 0x8:
    RegBase = AVR::R29R28;
    break;
  case 0x0:
    RegBase = AVR::R31R30;
    break;
  default:
    return MCDisassembler::Fail;
  }

  // Set the opcode.
  switch (Insn & 0x203) {
  case 0x200:
    Inst.setOpcode(AVR::STPtrRr);
    Inst.addOperand(MCOperand::createReg(RegBase));
    Inst.addOperand(MCOperand::createReg(RegVal));
    return MCDisassembler::Success;
  case 0x201:
    Inst.setOpcode(AVR::STPtrPiRr);
    break;
  case 0x202:
    Inst.setOpcode(AVR::STPtrPdRr);
    break;
  case 0:
    Inst.setOpcode(AVR::LDRdPtr);
    Inst.addOperand(MCOperand::createReg(RegVal));
    Inst.addOperand(MCOperand::createReg(RegBase));
    return MCDisassembler::Success;
  case 1:
    Inst.setOpcode(AVR::LDRdPtrPi);
    break;
  case 2:
    Inst.setOpcode(AVR::LDRdPtrPd);
    break;
  default:
    return MCDisassembler::Fail;
  }

  // Build postinc/predec machine instructions.
  if ((Insn & 0x200) == 0) { // This is a load instruction.
    Inst.addOperand(MCOperand::createReg(RegVal));
    Inst.addOperand(MCOperand::createReg(RegBase));
    Inst.addOperand(MCOperand::createReg(RegBase));
  } else { // This is a store instruction.
    Inst.addOperand(MCOperand::createReg(RegBase));
    Inst.addOperand(MCOperand::createReg(RegBase));
    Inst.addOperand(MCOperand::createReg(RegVal));
    // STPtrPiRr and STPtrPdRr have an extra immediate operand.
    Inst.addOperand(MCOperand::createImm(1));
  }

  return MCDisassembler::Success;
}

#include "AVRGenDisassemblerTables.inc"

static DecodeStatus readInstruction16(ArrayRef<uint8_t> Bytes, uint64_t Address,
                                      uint64_t &Size, uint32_t &Insn) {
  if (Bytes.size() < 2) {
    Size = 0;
    return MCDisassembler::Fail;
  }

  Size = 2;
  Insn = (Bytes[0] << 0) | (Bytes[1] << 8);

  return MCDisassembler::Success;
}

static DecodeStatus readInstruction32(ArrayRef<uint8_t> Bytes, uint64_t Address,
                                      uint64_t &Size, uint32_t &Insn) {

  if (Bytes.size() < 4) {
    Size = 0;
    return MCDisassembler::Fail;
  }

  Size = 4;
  Insn =
      (Bytes[0] << 16) | (Bytes[1] << 24) | (Bytes[2] << 0) | (Bytes[3] << 8);

  return MCDisassembler::Success;
}

static const uint8_t *getDecoderTable(uint64_t Size) {

  switch (Size) {
  case 2:
    return DecoderTable16;
  case 4:
    return DecoderTable32;
  default:
    llvm_unreachable("instructions must be 16 or 32-bits");
  }
}

DecodeStatus AVRDisassembler::getInstruction(MCInst &Instr, uint64_t &Size,
                                             ArrayRef<uint8_t> Bytes,
                                             uint64_t Address,
                                             raw_ostream &CStream) const {
  uint32_t Insn;

  DecodeStatus Result;

  // Try decode a 16-bit instruction.
  {
    Result = readInstruction16(Bytes, Address, Size, Insn);

    if (Result == MCDisassembler::Fail)
      return MCDisassembler::Fail;

    // Try to decode AVRTiny instructions.
    if (STI.hasFeature(AVR::FeatureTinyEncoding)) {
      Result = decodeInstruction(DecoderTableAVRTiny16, Instr, Insn, Address,
                                 this, STI);
      if (Result != MCDisassembler::Fail)
        return Result;
    }

    // Try to auto-decode a 16-bit instruction.
    Result = decodeInstruction(getDecoderTable(Size), Instr, Insn, Address,
                               this, STI);
    if (Result != MCDisassembler::Fail)
      return Result;

    // Try to decode to a load/store instruction. ST/LD need a specified
    // DecoderMethod, as they already have a specified PostEncoderMethod.
    Result = decodeLoadStore(Instr, Insn, Address, this);
    if (Result != MCDisassembler::Fail)
      return Result;
  }

  // Try decode a 32-bit instruction.
  {
    Result = readInstruction32(Bytes, Address, Size, Insn);

    if (Result == MCDisassembler::Fail)
      return MCDisassembler::Fail;

    Result = decodeInstruction(getDecoderTable(Size), Instr, Insn, Address,
                               this, STI);

    if (Result != MCDisassembler::Fail) {
      return Result;
    }

    return MCDisassembler::Fail;
  }
}

typedef DecodeStatus (*DecodeFunc)(MCInst &MI, unsigned insn, uint64_t Address,
                                   const MCDisassembler *Decoder);
