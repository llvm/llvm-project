//===-- MSP430Disassembler.cpp - Disassembler for MSP430 ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSP430Disassembler class.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MSP430MCTargetDesc.h"
#include "MSP430.h"
#include "TargetInfo/MSP430TargetInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDecoder.h"
#include "llvm/MC/MCDecoderOps.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::MCD;

#define DEBUG_TYPE "msp430-disassembler"

typedef MCDisassembler::DecodeStatus DecodeStatus;

namespace {
class MSP430Disassembler : public MCDisassembler {
  DecodeStatus getInstructionI(MCInst &MI, uint64_t &Size,
                               ArrayRef<uint8_t> Bytes, uint64_t Address,
                               raw_ostream &CStream) const;

  DecodeStatus getInstructionII(MCInst &MI, uint64_t &Size,
                                ArrayRef<uint8_t> Bytes, uint64_t Address,
                                raw_ostream &CStream) const;

  DecodeStatus getInstructionCJ(MCInst &MI, uint64_t &Size,
                                ArrayRef<uint8_t> Bytes, uint64_t Address,
                                raw_ostream &CStream) const;

  DecodeStatus getInstructionX(MCInst &MI, uint64_t &Size,
                               ArrayRef<uint8_t> Bytes, uint64_t Address,
                               raw_ostream &CStream) const;

public:
  MSP430Disassembler(const MCSubtargetInfo &STI, MCContext &Ctx)
      : MCDisassembler(STI, Ctx) {}

  DecodeStatus getInstruction(MCInst &MI, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &CStream) const override;
};
} // end anonymous namespace

static MCDisassembler *createMSP430Disassembler(const Target &T,
                                                const MCSubtargetInfo &STI,
                                                MCContext &Ctx) {
  return new MSP430Disassembler(STI, Ctx);
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeMSP430Disassembler() {
  TargetRegistry::RegisterMCDisassembler(getTheMSP430Target(),
                                         createMSP430Disassembler);
}

static const unsigned GR8DecoderTable[] = {
  MSP430::PCB,  MSP430::SPB,  MSP430::SRB,  MSP430::CGB,
  MSP430::R4B,  MSP430::R5B,  MSP430::R6B,  MSP430::R7B,
  MSP430::R8B,  MSP430::R9B,  MSP430::R10B, MSP430::R11B,
  MSP430::R12B, MSP430::R13B, MSP430::R14B, MSP430::R15B
};

static DecodeStatus DecodeGR8RegisterClass(MCInst &MI, uint64_t RegNo,
                                           uint64_t Address,
                                           const MCDisassembler *Decoder) {
  if (RegNo > 15)
    return MCDisassembler::Fail;

  unsigned Reg = GR8DecoderTable[RegNo];
  MI.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static const unsigned GR16DecoderTable[] = {
  MSP430::PC,  MSP430::SP,  MSP430::SR,  MSP430::CG,
  MSP430::R4,  MSP430::R5,  MSP430::R6,  MSP430::R7,
  MSP430::R8,  MSP430::R9,  MSP430::R10, MSP430::R11,
  MSP430::R12, MSP430::R13, MSP430::R14, MSP430::R15
};

static DecodeStatus DecodeGR16RegisterClass(MCInst &MI, uint64_t RegNo,
                                            uint64_t Address,
                                            const MCDisassembler *Decoder) {
  if (RegNo > 15)
    return MCDisassembler::Fail;

  unsigned Reg = GR16DecoderTable[RegNo];
  MI.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeCGImm(MCInst &MI, uint64_t Bits, uint64_t Address,
                                const MCDisassembler *Decoder) {
  int64_t Imm;
  switch (Bits) {
  default:
    llvm_unreachable("Invalid immediate value");
  case 0x22: Imm =  4; break;
  case 0x32: Imm =  8; break;
  case 0x03: Imm =  0; break;
  case 0x13: Imm =  1; break;
  case 0x23: Imm =  2; break;
  case 0x33: Imm = -1; break;
  }
  MI.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

/// Decode the 2-bit shift count of RRCM, RRAM, RRUM and RLAM. It is stored as
/// count - 1.
static DecodeStatus decodeRpt2Imm(MCInst &MI, uint64_t Bits, uint64_t Address,
                                  const MCDisassembler *Decoder) {
  int64_t Imm = Bits + 1;
  MI.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeMemOperand(MCInst &MI, uint64_t Bits,
                                     uint64_t Address,
                                     const MCDisassembler *Decoder) {
  unsigned Reg = Bits & 15;
  unsigned Imm = Bits >> 4;

  if (DecodeGR16RegisterClass(MI, Reg, Address, Decoder) !=
      MCDisassembler::Success)
    return MCDisassembler::Fail;
  
  MI.addOperand(MCOperand::createImm((int16_t)Imm));
  return MCDisassembler::Success;
}

#include "MSP430GenDisassemblerTables.inc"

enum AddrMode {
  amInvalid = 0,
  amRegister,
  amIndexed,
  amIndirect,
  amIndirectPost,
  amSymbolic,
  amImmediate,
  amAbsolute,
  amConstant
};

static AddrMode DecodeSrcAddrMode(unsigned Rs, unsigned As) {
  switch (Rs) {
  case 0:
    if (As == 1) return amSymbolic;
    if (As == 2) return amInvalid;
    if (As == 3) return amImmediate;
    break;
  case 2:
    if (As == 1) return amAbsolute;
    if (As == 2) return amConstant;
    if (As == 3) return amConstant;
    break;
  case 3:
    return amConstant;
  default:
    break;
  }
  switch (As) {
  case 0: return amRegister;
  case 1: return amIndexed;
  case 2: return amIndirect;
  case 3: return amIndirectPost;
  default:
    llvm_unreachable("As out of range");
  }
}

static AddrMode DecodeSrcAddrModeI(unsigned Insn) {
  unsigned Rs = fieldFromInstruction(Insn, 8, 4);
  unsigned As = fieldFromInstruction(Insn, 4, 2);
  return DecodeSrcAddrMode(Rs, As);
}

static AddrMode DecodeSrcAddrModeII(unsigned Insn) {
  unsigned Rs = fieldFromInstruction(Insn, 0, 4);
  unsigned As = fieldFromInstruction(Insn, 4, 2);
  return DecodeSrcAddrMode(Rs, As);
}

static AddrMode DecodeDstAddrMode(unsigned Insn) {
  unsigned Rd = fieldFromInstruction(Insn, 0, 4);
  unsigned Ad = fieldFromInstruction(Insn, 7, 1);
  switch (Rd) {
  case 0: return Ad ? amSymbolic : amRegister;
  case 2: return Ad ? amAbsolute : amRegister;
  default:
    break;
  }
  return Ad ? amIndexed : amRegister;
}

static const uint8_t *getDecoderTable(AddrMode SrcAM, unsigned Words) {
  assert(0 < Words && Words < 4 && "Incorrect number of words");
  switch (SrcAM) {
  default:
    llvm_unreachable("Invalid addressing mode");
  case amRegister:
    assert(Words < 3 && "Incorrect number of words");
    return Words == 2 ? DecoderTableAlpha32 : DecoderTableAlpha16;
  case amConstant:
    assert(Words < 3 && "Incorrect number of words");
    return Words == 2 ? DecoderTableBeta32 : DecoderTableBeta16;
  case amIndexed:
  case amSymbolic:
  case amImmediate:
  case amAbsolute:
    assert(Words > 1 && "Incorrect number of words");
    return Words == 2 ? DecoderTableGamma32 : DecoderTableGamma48;
  case amIndirect:
  case amIndirectPost:
    assert(Words < 3 && "Incorrect number of words");
    return Words == 2 ? DecoderTableDelta32 : DecoderTableDelta16;
  }
}

DecodeStatus MSP430Disassembler::getInstructionI(MCInst &MI, uint64_t &Size,
                                                 ArrayRef<uint8_t> Bytes,
                                                 uint64_t Address,
                                                 raw_ostream &CStream) const {
  uint64_t Insn = support::endian::read16le(Bytes.data());
  AddrMode SrcAM = DecodeSrcAddrModeI(Insn);
  AddrMode DstAM = DecodeDstAddrMode(Insn);
  if (SrcAM == amInvalid || DstAM == amInvalid) {
    Size = 2; // skip one word and let disassembler to try further
    return MCDisassembler::Fail;
  }

  unsigned Words = 1;
  switch (SrcAM) {
  case amIndexed:
  case amSymbolic:
  case amImmediate:
  case amAbsolute:
    if (Bytes.size() < (Words + 1) * 2) {
      Size = 2;
      return DecodeStatus::Fail;
    }
    Insn |= (uint64_t)support::endian::read16le(Bytes.data() + 2) << 16;
    ++Words;
    break;
  default:
    break;
  }
  switch (DstAM) {
  case amIndexed:
  case amSymbolic:
  case amAbsolute:
    if (Bytes.size() < (Words + 1) * 2) {
      Size = 2;
      return DecodeStatus::Fail;
    }
    Insn |= (uint64_t)support::endian::read16le(Bytes.data() + Words * 2)
        << (Words * 16);
    ++Words;
    break;
  default:
    break;
  }

  DecodeStatus Result = decodeInstruction(getDecoderTable(SrcAM, Words), MI,
                                          Insn, Address, this, STI);
  if (Result != MCDisassembler::Fail) {
    Size = Words * 2;
    return Result;
  }

  Size = 2;
  return DecodeStatus::Fail;
}

DecodeStatus MSP430Disassembler::getInstructionII(MCInst &MI, uint64_t &Size,
                                                  ArrayRef<uint8_t> Bytes,
                                                  uint64_t Address,
                                                  raw_ostream &CStream) const {
  uint64_t Insn = support::endian::read16le(Bytes.data());
  AddrMode SrcAM = DecodeSrcAddrModeII(Insn);
  if (SrcAM == amInvalid) {
    Size = 2; // skip one word and let disassembler to try further
    return MCDisassembler::Fail;
  }

  unsigned Words = 1;
  switch (SrcAM) {
  case amIndexed:
  case amSymbolic:
  case amImmediate:
  case amAbsolute:
    if (Bytes.size() < (Words + 1) * 2) {
      Size = 2;
      return DecodeStatus::Fail;
    }
    Insn |= (uint64_t)support::endian::read16le(Bytes.data() + 2) << 16;
    ++Words;
    break;
  default:
    break;
  }

  const uint8_t *DecoderTable = Words == 2 ? DecoderTable32 : DecoderTable16;
  DecodeStatus Result = decodeInstruction(DecoderTable, MI, Insn, Address,
                                          this, STI);
  if (Result != MCDisassembler::Fail) {
    Size = Words * 2;
    return Result;
  }

  Size = 2;
  return DecodeStatus::Fail;
}

static MSP430CC::CondCodes getCondCode(unsigned Cond) {
  switch (Cond) {
  case 0: return MSP430CC::COND_NE;
  case 1: return MSP430CC::COND_E;
  case 2: return MSP430CC::COND_LO;
  case 3: return MSP430CC::COND_HS;
  case 4: return MSP430CC::COND_N;
  case 5: return MSP430CC::COND_GE;
  case 6: return MSP430CC::COND_L;
  case 7: return MSP430CC::COND_NONE;
  default:
    llvm_unreachable("Cond out of range");
  }
}

DecodeStatus MSP430Disassembler::getInstructionCJ(MCInst &MI, uint64_t &Size,
                                                  ArrayRef<uint8_t> Bytes,
                                                  uint64_t Address,
                                                  raw_ostream &CStream) const {
  uint64_t Insn = support::endian::read16le(Bytes.data());
  unsigned Cond = fieldFromInstruction(Insn, 10, 3);
  unsigned Offset = fieldFromInstruction(Insn, 0, 10);

  MI.addOperand(MCOperand::createImm(SignExtend32(Offset, 10)));

  if (Cond == 7)
    MI.setOpcode(MSP430::JMP);
  else {
    MI.setOpcode(MSP430::JCC);
    MI.addOperand(MCOperand::createImm(getCondCode(Cond)));
  }

  Size = 2;
  return DecodeStatus::Success;
}

static bool isExtensionWord(uint64_t Word) { return (Word & 0xf800) == 0x1800; }

DecodeStatus MSP430Disassembler::getInstructionX(MCInst &MI, uint64_t &Size,
                                                 ArrayRef<uint8_t> Bytes,
                                                 uint64_t Address,
                                                 raw_ostream &CStream) const {
  uint64_t Insn = support::endian::read16le(Bytes.data());
  unsigned Words = 1;
  if (isExtensionWord(Insn)) {
    if (Bytes.size() < 4) {
      Size = 2;
      return DecodeStatus::Fail;
    }
    Insn |= (uint64_t)support::endian::read16le(Bytes.data() + 2) << 16;
    Words = 2;
  }

  // MSP430X32 decodes the extension word together with the following Format I
  // or II word. Each combination is its own instruction, like RRCX_16 or
  // ADDX16rr, so the plain Format I and II tables are not used here.
  const uint8_t *DecoderTable =
      Words == 2 ? DecoderTableMSP430X32 : DecoderTableMSP430X16;
  DecodeStatus Result =
      decodeInstruction(DecoderTable, MI, Insn, Address, this, STI);
  // The word after an extension word is a valid Format I or II instruction on
  // its own. Skip it on failure too, so it is not shown without the extension
  // word.
  Size = Words * 2;
  return Result;
}

DecodeStatus MSP430Disassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                                ArrayRef<uint8_t> Bytes,
                                                uint64_t Address,
                                                raw_ostream &CStream) const {
  if (Bytes.size() < 2) {
    Size = 0;
    return MCDisassembler::Fail;
  }

  // The first word selects the instruction format:
  //   0x0000-0x0fff  MSP430X address instructions
  //   0x1000-0x17ff  Format II
  //   0x1800-0x1fff  MSP430X extension word, followed by a Format I or II word
  //   0x2000-0x3fff  Jumps
  //   0x4000-0xffff  Format I
  uint64_t Insn = support::endian::read16le(Bytes.data());
  unsigned FormatBits = fieldFromInstruction(Insn, 13, 3);
  switch (FormatBits) {
  case 0:
    if (!(Insn & 0x1000) || isExtensionWord(Insn))
      return getInstructionX(MI, Size, Bytes, Address, CStream);
    return getInstructionII(MI, Size, Bytes, Address, CStream);
  case 1:
    return getInstructionCJ(MI, Size, Bytes, Address, CStream);
  default:
    return getInstructionI(MI, Size, Bytes, Address, CStream);
  }
}
