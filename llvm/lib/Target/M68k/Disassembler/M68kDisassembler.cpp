//===-- M68kDisassembler.cpp - Disassembler for M68k ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is part of the M68k Disassembler.
//
//===----------------------------------------------------------------------===//

#include "M68k.h"
#include "M68kRegisterInfo.h"
#include "M68kSubtarget.h"
#include "MCTargetDesc/M68kMCCodeEmitter.h"
#include "MCTargetDesc/M68kMCTargetDesc.h"
#include "TargetInfo/M68kTargetInfo.h"

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCDecoderOps.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#define DEBUG_TYPE "m68k-disassembler"

typedef MCDisassembler::DecodeStatus DecodeStatus;
const unsigned MaxInstrWord = 11;

static const unsigned RegisterDecode[] = {
    M68k::D0, M68k::D1, M68k::D2, M68k::D3, M68k::D4, M68k::D5,
    M68k::D6, M68k::D7, M68k::A0, M68k::A1, M68k::A2, M68k::A3,
    M68k::A4, M68k::A5, M68k::A6, M68k::SP,
};

static DecodeStatus DecodeRegisterClass(MCInst &Inst, uint64_t RegNo,
                                        uint64_t Address, const void *Decoder) {
  if (RegNo >= 16)
    return DecodeStatus::Fail;
  Inst.addOperand(MCOperand::createReg(RegisterDecode[RegNo]));
  return DecodeStatus::Success;
}

static DecodeStatus DecodeDR32RegisterClass(MCInst &Inst, uint64_t RegNo,
                                            uint64_t Address,
                                            const void *Decoder) {
  return DecodeRegisterClass(Inst, RegNo, Address, Decoder);
}

static DecodeStatus DecodeDR16RegisterClass(MCInst &Inst, uint64_t RegNo,
                                            uint64_t Address,
                                            const void *Decoder) {
  return DecodeRegisterClass(Inst, RegNo, Address, Decoder);
}

static DecodeStatus DecodeDR8RegisterClass(MCInst &Inst, uint64_t RegNo,
                                           uint64_t Address,
                                           const void *Decoder) {
  return DecodeRegisterClass(Inst, RegNo, Address, Decoder);
}

static DecodeStatus DecodeAR32RegisterClass(MCInst &Inst, uint64_t RegNo,
                                            uint64_t Address,
                                            const void *Decoder) {
  return DecodeRegisterClass(Inst, RegNo | 8ULL, Address, Decoder);
}

static DecodeStatus DecodeAR16RegisterClass(MCInst &Inst, uint64_t RegNo,
                                            uint64_t Address,
                                            const void *Decoder) {
  return DecodeRegisterClass(Inst, RegNo | 8ULL, Address, Decoder);
}

static DecodeStatus DecodeXR32RegisterClass(MCInst &Inst, uint64_t RegNo,
                                            uint64_t Address,
                                            const void *Decoder) {
  return DecodeRegisterClass(Inst, RegNo, Address, Decoder);
}

static DecodeStatus DecodeXR16RegisterClass(MCInst &Inst, uint64_t RegNo,
                                            uint64_t Address,
                                            const void *Decoder) {
  return DecodeRegisterClass(Inst, RegNo, Address, Decoder);
}

static DecodeStatus DecodeCCRCRegisterClass(MCInst &Inst, APInt &Insn,
                                            uint64_t Address,
                                            const void *Decoder) {
  llvm_unreachable("unimplemented");
}

#include "M68kGenDisassemblerTable.inc"

/// A disassembler class for M68k.
class M68kDisassembler : public MCDisassembler {
  MCInstrInfo *MCII;

public:
  M68kDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx,
                   MCInstrInfo *MCII)
      : MCDisassembler(STI, Ctx), MCII(MCII) {}
  virtual ~M68kDisassembler() {}

  DecodeStatus getInstruction(MCInst &Instr, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &CStream) const override;
};

DecodeStatus M68kDisassembler::getInstruction(MCInst &Instr, uint64_t &Size,
                                              ArrayRef<uint8_t> Bytes,
                                              uint64_t Address,
                                              raw_ostream &CStream) const {
  DecodeStatus Result;
  auto MakeUp = [&](APInt &Insn, unsigned InstrBits) {
    unsigned Idx = Insn.getBitWidth() >> 3;
    unsigned RoundUp = alignTo(InstrBits, Align(16));
    Insn = Insn.zextOrSelf(RoundUp);
    RoundUp = RoundUp >> 3;
    for (; Idx < RoundUp; Idx += 2) {
      Insn.insertBits(support::endian::read16be(&Bytes[Idx]), Idx * 8, 16);
    }
  };
  APInt Insn(16, support::endian::read16be(Bytes.data()));
  // 2 bytes of data are consumed, so set Size to 2
  // If we don't do this, disassembler may generate result even
  // the encoding is invalid. We need to let it fail correctly.
  Size = 2;
  Result = decodeInstruction(DecoderTable80, Instr, Insn, Address, this, STI,
                             MakeUp);
  if (Result == DecodeStatus::Success)
    Size = InstrLenTable[Instr.getOpcode()] >> 3;
  return Result;
}

static MCDisassembler *createM68kDisassembler(const Target &T,
                                              const MCSubtargetInfo &STI,
                                              MCContext &Ctx) {
  return new M68kDisassembler(STI, Ctx, T.createMCInstrInfo());
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeM68kDisassembler() {
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(getTheM68kTarget(),
                                         createM68kDisassembler);
}
