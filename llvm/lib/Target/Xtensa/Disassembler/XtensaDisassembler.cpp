//===-- XtensaDisassembler.cpp - Disassembler for Xtensa ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the XtensaDisassembler class.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/XtensaMCTargetDesc.h"
#include "TargetInfo/XtensaTargetInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDecoderOps.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Endian.h"

using namespace llvm;

#define DEBUG_TYPE "Xtensa-disassembler"

using DecodeStatus = MCDisassembler::DecodeStatus;

namespace {

class XtensaDisassembler : public MCDisassembler {
  bool IsLittleEndian;

public:
  XtensaDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx, bool isLE)
      : MCDisassembler(STI, Ctx), IsLittleEndian(isLE) {}

  bool hasDensity() const { return STI.hasFeature(Xtensa::FeatureDensity); }

  DecodeStatus getInstruction(MCInst &Instr, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &CStream) const override;
};
} // end anonymous namespace

static MCDisassembler *createXtensaDisassembler(const Target &T,
                                                const MCSubtargetInfo &STI,
                                                MCContext &Ctx) {
  return new XtensaDisassembler(STI, Ctx, true);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeXtensaDisassembler() {
  TargetRegistry::RegisterMCDisassembler(getTheXtensaTarget(),
                                         createXtensaDisassembler);
}

const MCPhysReg ARDecoderTable[] = {
    Xtensa::A0,  Xtensa::SP,  Xtensa::A2,  Xtensa::A3, Xtensa::A4,  Xtensa::A5,
    Xtensa::A6,  Xtensa::A7,  Xtensa::A8,  Xtensa::A9, Xtensa::A10, Xtensa::A11,
    Xtensa::A12, Xtensa::A13, Xtensa::A14, Xtensa::A15};

static DecodeStatus DecodeARRegisterClass(MCInst &Inst, uint64_t RegNo,
                                          uint64_t Address,
                                          const void *Decoder) {
  if (RegNo >= std::size(ARDecoderTable))
    return MCDisassembler::Fail;

  MCPhysReg Reg = ARDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeMRRegisterClass(MCInst &Inst, uint64_t RegNo,
                                          uint64_t Address,
                                          const void *Decoder) {
  if (RegNo > 3)
    return MCDisassembler::Fail;

  MCPhysReg Reg = Xtensa::M0 + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeMR01RegisterClass(MCInst &Inst, uint64_t RegNo,
                                            uint64_t Address,
                                            const void *Decoder) {
  if (RegNo > 1)
    return MCDisassembler::Fail;

  MCPhysReg Reg = Xtensa::M0 + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeMR23RegisterClass(MCInst &Inst, uint64_t RegNo,
                                            uint64_t Address,
                                            const void *Decoder) {
  if (RegNo > 1)
    return MCDisassembler::Fail;

  MCPhysReg Reg = Xtensa::M2 + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeFPRRegisterClass(MCInst &Inst, uint64_t RegNo,
                                           uint64_t Address,
                                           const void *Decoder) {
  if (RegNo > 15)
    return MCDisassembler::Fail;

  MCPhysReg Reg = Xtensa::F0 + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeURRegisterClass(MCInst &Inst, uint64_t RegNo,
                                          uint64_t Address,
                                          const MCDisassembler *Decoder) {
  if (RegNo > 255)
    return MCDisassembler::Fail;

  Xtensa::RegisterAccessType RAType = Inst.getOpcode() == Xtensa::WUR
                                          ? Xtensa::REGISTER_WRITE
                                          : Xtensa::REGISTER_READ;

  const XtensaDisassembler *Dis =
      static_cast<const XtensaDisassembler *>(Decoder);
  const MCRegisterInfo *MRI = Dis->getContext().getRegisterInfo();
  MCPhysReg Reg = Xtensa::getUserRegister(RegNo, *MRI);
  if (!Xtensa::checkRegister(Reg, Decoder->getSubtargetInfo().getFeatureBits(),
                             RAType))
    return MCDisassembler::Fail;

  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

struct DecodeRegister {
  MCPhysReg Reg;
  uint32_t RegNo;
};

const DecodeRegister SRDecoderTable[] = {
    {Xtensa::LBEG, 0},         {Xtensa::LEND, 1},
    {Xtensa::LCOUNT, 2},       {Xtensa::SAR, 3},
    {Xtensa::BREG, 4},         {Xtensa::LITBASE, 5},
    {Xtensa::SCOMPARE1, 12},   {Xtensa::ACCLO, 16},
    {Xtensa::ACCHI, 17},       {Xtensa::M0, 32},
    {Xtensa::M1, 33},          {Xtensa::M2, 34},
    {Xtensa::M3, 35},          {Xtensa::WINDOWBASE, 72},
    {Xtensa::WINDOWSTART, 73}, {Xtensa::IBREAKENABLE, 96},
    {Xtensa::MEMCTL, 97},      {Xtensa::ATOMCTL, 99},
    {Xtensa::DDR, 104},        {Xtensa::IBREAKA0, 128},
    {Xtensa::IBREAKA1, 129},   {Xtensa::DBREAKA0, 144},
    {Xtensa::DBREAKA1, 145},   {Xtensa::DBREAKC0, 160},
    {Xtensa::DBREAKC1, 161},   {Xtensa::CONFIGID0, 176},
    {Xtensa::EPC1, 177},       {Xtensa::EPC2, 178},
    {Xtensa::EPC3, 179},       {Xtensa::EPC4, 180},
    {Xtensa::EPC5, 181},       {Xtensa::EPC6, 182},
    {Xtensa::EPC7, 183},       {Xtensa::DEPC, 192},
    {Xtensa::EPS2, 194},       {Xtensa::EPS3, 195},
    {Xtensa::EPS4, 196},       {Xtensa::EPS5, 197},
    {Xtensa::EPS6, 198},       {Xtensa::EPS7, 199},
    {Xtensa::CONFIGID1, 208},  {Xtensa::EXCSAVE1, 209},
    {Xtensa::EXCSAVE2, 210},   {Xtensa::EXCSAVE3, 211},
    {Xtensa::EXCSAVE4, 212},   {Xtensa::EXCSAVE5, 213},
    {Xtensa::EXCSAVE6, 214},   {Xtensa::EXCSAVE7, 215},
    {Xtensa::CPENABLE, 224},   {Xtensa::INTERRUPT, 226},
    {Xtensa::INTCLEAR, 227},   {Xtensa::INTENABLE, 228},
    {Xtensa::PS, 230},         {Xtensa::VECBASE, 231},
    {Xtensa::EXCCAUSE, 232},   {Xtensa::DEBUGCAUSE, 233},
    {Xtensa::CCOUNT, 234},     {Xtensa::PRID, 235},
    {Xtensa::ICOUNT, 236},     {Xtensa::ICOUNTLEVEL, 237},
    {Xtensa::EXCVADDR, 238},   {Xtensa::CCOMPARE0, 240},
    {Xtensa::CCOMPARE1, 241},  {Xtensa::CCOMPARE2, 242},
    {Xtensa::MISC0, 244},      {Xtensa::MISC1, 245},
    {Xtensa::MISC2, 246},      {Xtensa::MISC3, 247}};

static DecodeStatus DecodeSRRegisterClass(MCInst &Inst, uint64_t RegNo,
                                          uint64_t Address,
                                          const MCDisassembler *Decoder) {
  if (RegNo > 255)
    return MCDisassembler::Fail;

  Xtensa::RegisterAccessType RAType =
      Inst.getOpcode() == Xtensa::WSR
          ? Xtensa::REGISTER_WRITE
          : (Inst.getOpcode() == Xtensa::RSR ? Xtensa::REGISTER_READ
                                             : Xtensa::REGISTER_EXCHANGE);

  for (unsigned i = 0; i < std::size(SRDecoderTable); i++) {
    if (SRDecoderTable[i].RegNo == RegNo) {
      MCPhysReg Reg = SRDecoderTable[i].Reg;

      // Handle special case. The INTERRUPT/INTSET registers use the same
      // encoding, but INTERRUPT used for read and INTSET for write.
      if (Reg == Xtensa::INTERRUPT && RAType == Xtensa::REGISTER_WRITE) {
        Reg = Xtensa::INTSET;
      }

      if (!Xtensa::checkRegister(
              Reg, Decoder->getSubtargetInfo().getFeatureBits(), RAType))
        return MCDisassembler::Fail;

      Inst.addOperand(MCOperand::createReg(Reg));
      return MCDisassembler::Success;
    }
  }

  return MCDisassembler::Fail;
}

static DecodeStatus DecodeBRRegisterClass(MCInst &Inst, uint64_t RegNo,
                                          uint64_t Address,
                                          const void *Decoder) {
  if (RegNo > 15)
    return MCDisassembler::Fail;

  MCPhysReg Reg = Xtensa::B0 + RegNo;
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static bool tryAddingSymbolicOperand(int64_t Value, bool isBranch,
                                     uint64_t Address, uint64_t Offset,
                                     uint64_t InstSize, MCInst &MI,
                                     const void *Decoder) {
  const MCDisassembler *Dis = static_cast<const MCDisassembler *>(Decoder);
  return Dis->tryAddingSymbolicOperand(MI, Value, Address, isBranch, Offset,
                                       /*OpSize=*/0, InstSize);
}

static DecodeStatus decodeCallOperand(MCInst &Inst, uint64_t Imm,
                                      int64_t Address, const void *Decoder) {
  assert(isUInt<18>(Imm) && "Invalid immediate");
  Inst.addOperand(
      MCOperand::createImm(SignExtend64<20>(Imm << 2) + (Address & 0x3)));
  return MCDisassembler::Success;
}

static DecodeStatus decodeJumpOperand(MCInst &Inst, uint64_t Imm,
                                      int64_t Address, const void *Decoder) {
  assert(isUInt<18>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(SignExtend64<18>(Imm)));
  return MCDisassembler::Success;
}

static DecodeStatus decodeBranchOperand(MCInst &Inst, uint64_t Imm,
                                        int64_t Address, const void *Decoder) {
  switch (Inst.getOpcode()) {
  case Xtensa::BEQZ:
  case Xtensa::BGEZ:
  case Xtensa::BLTZ:
  case Xtensa::BNEZ:
    assert(isUInt<12>(Imm) && "Invalid immediate");
    if (!tryAddingSymbolicOperand(SignExtend64<12>(Imm) + 4 + Address, true,
                                  Address, 0, 3, Inst, Decoder))
      Inst.addOperand(MCOperand::createImm(SignExtend64<12>(Imm)));
    break;
  default:
    assert(isUInt<8>(Imm) && "Invalid immediate");
    if (!tryAddingSymbolicOperand(SignExtend64<8>(Imm) + 4 + Address, true,
                                  Address, 0, 3, Inst, Decoder))
      Inst.addOperand(MCOperand::createImm(SignExtend64<8>(Imm)));
  }
  return MCDisassembler::Success;
}

static DecodeStatus decodeLoopOperand(MCInst &Inst, uint64_t Imm,
                                      int64_t Address, const void *Decoder) {

  assert(isUInt<8>(Imm) && "Invalid immediate");
  if (!tryAddingSymbolicOperand(Imm + 4 + Address, true, Address, 0, 3, Inst,
                                Decoder))
    Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeL32ROperand(MCInst &Inst, uint64_t Imm,
                                      int64_t Address, const void *Decoder) {

  assert(isUInt<16>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(
      SignExtend64<17>((Imm << 2) + 0x40000 + (Address & 0x3))));
  return MCDisassembler::Success;
}

static DecodeStatus decodeImm8Operand(MCInst &Inst, uint64_t Imm,
                                      int64_t Address, const void *Decoder) {
  assert(isUInt<8>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(SignExtend64<8>(Imm)));
  return MCDisassembler::Success;
}

static DecodeStatus decodeImm8_sh8Operand(MCInst &Inst, uint64_t Imm,
                                          int64_t Address,
                                          const void *Decoder) {
  assert(isUInt<8>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(SignExtend64<16>(Imm << 8)));
  return MCDisassembler::Success;
}

static DecodeStatus decodeImm12Operand(MCInst &Inst, uint64_t Imm,
                                       int64_t Address, const void *Decoder) {
  assert(isUInt<12>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(SignExtend64<12>(Imm)));
  return MCDisassembler::Success;
}

static DecodeStatus decodeUimm4Operand(MCInst &Inst, uint64_t Imm,
                                       int64_t Address, const void *Decoder) {
  assert(isUInt<4>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeUimm5Operand(MCInst &Inst, uint64_t Imm,
                                       int64_t Address, const void *Decoder) {
  assert(isUInt<5>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeImm1_16Operand(MCInst &Inst, uint64_t Imm,
                                         int64_t Address, const void *Decoder) {
  assert(isUInt<4>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(Imm + 1));
  return MCDisassembler::Success;
}

static DecodeStatus decodeImm1n_15Operand(MCInst &Inst, uint64_t Imm,
                                          int64_t Address,
                                          const void *Decoder) {
  assert(isUInt<4>(Imm) && "Invalid immediate");
  if (!Imm)
    Inst.addOperand(MCOperand::createImm(-1));
  else
    Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeImm32n_95Operand(MCInst &Inst, uint64_t Imm,
                                           int64_t Address,
                                           const void *Decoder) {
  assert(isUInt<7>(Imm) && "Invalid immediate");
  if ((Imm & 0x60) == 0x60)
    Inst.addOperand(MCOperand::createImm((~0x1f) | Imm));
  else
    Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeImm8n_7Operand(MCInst &Inst, uint64_t Imm,
                                         int64_t Address, const void *Decoder) {
  assert(isUInt<4>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(Imm > 7 ? Imm - 16 : Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeImm64n_4nOperand(MCInst &Inst, uint64_t Imm,
                                           int64_t Address,
                                           const void *Decoder) {
  assert(isUInt<6>(Imm) && ((Imm & 0x3) == 0) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm((~0x3f) | (Imm)));
  return MCDisassembler::Success;
}

static DecodeStatus decodeEntry_Imm12OpValue(MCInst &Inst, uint64_t Imm,
                                             int64_t Address,
                                             const void *Decoder) {
  assert(isUInt<15>(Imm) && ((Imm & 0x7) == 0) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(Imm));
  return MCDisassembler::Success;
}

static DecodeStatus decodeShimm1_31Operand(MCInst &Inst, uint64_t Imm,
                                           int64_t Address,
                                           const void *Decoder) {
  assert(isUInt<5>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(32 - Imm));
  return MCDisassembler::Success;
}

static int64_t TableB4const[16] = {-1, 1,  2,  3,  4,  5,  6,   7,
                                   8,  10, 12, 16, 32, 64, 128, 256};
static DecodeStatus decodeB4constOperand(MCInst &Inst, uint64_t Imm,
                                         int64_t Address, const void *Decoder) {
  assert(isUInt<4>(Imm) && "Invalid immediate");

  Inst.addOperand(MCOperand::createImm(TableB4const[Imm]));
  return MCDisassembler::Success;
}

static int64_t TableB4constu[16] = {32768, 65536, 2,  3,  4,  5,  6,   7,
                                    8,     10,    12, 16, 32, 64, 128, 256};
static DecodeStatus decodeB4constuOperand(MCInst &Inst, uint64_t Imm,
                                          int64_t Address,
                                          const void *Decoder) {
  assert(isUInt<4>(Imm) && "Invalid immediate");

  Inst.addOperand(MCOperand::createImm(TableB4constu[Imm]));
  return MCDisassembler::Success;
}

static DecodeStatus decodeImm7_22Operand(MCInst &Inst, uint64_t Imm,
                                         int64_t Address, const void *Decoder) {
  assert(isUInt<4>(Imm) && "Invalid immediate");
  Inst.addOperand(MCOperand::createImm(Imm + 7));
  return MCDisassembler::Success;
}

static DecodeStatus decodeMem8Operand(MCInst &Inst, uint64_t Imm,
                                      int64_t Address, const void *Decoder) {
  assert(isUInt<12>(Imm) && "Invalid immediate");
  DecodeARRegisterClass(Inst, Imm & 0xf, Address, Decoder);
  Inst.addOperand(MCOperand::createImm((Imm >> 4) & 0xff));
  return MCDisassembler::Success;
}

static DecodeStatus decodeMem16Operand(MCInst &Inst, uint64_t Imm,
                                       int64_t Address, const void *Decoder) {
  assert(isUInt<12>(Imm) && "Invalid immediate");
  DecodeARRegisterClass(Inst, Imm & 0xf, Address, Decoder);
  Inst.addOperand(MCOperand::createImm((Imm >> 3) & 0x1fe));
  return MCDisassembler::Success;
}

static DecodeStatus decodeMem32Operand(MCInst &Inst, uint64_t Imm,
                                       int64_t Address, const void *Decoder) {
  assert(isUInt<12>(Imm) && "Invalid immediate");
  DecodeARRegisterClass(Inst, Imm & 0xf, Address, Decoder);
  Inst.addOperand(MCOperand::createImm((Imm >> 2) & 0x3fc));
  return MCDisassembler::Success;
}

static DecodeStatus decodeMem32nOperand(MCInst &Inst, uint64_t Imm,
                                        int64_t Address, const void *Decoder) {
  assert(isUInt<8>(Imm) && "Invalid immediate");
  DecodeARRegisterClass(Inst, Imm & 0xf, Address, Decoder);
  Inst.addOperand(MCOperand::createImm((Imm >> 2) & 0x3c));
  return MCDisassembler::Success;
}

/// Read two bytes from the ArrayRef and return 16 bit data sorted
/// according to the given endianness.
static DecodeStatus readInstruction16(ArrayRef<uint8_t> Bytes, uint64_t Address,
                                      uint64_t &Size, uint64_t &Insn,
                                      bool IsLittleEndian) {
  // We want to read exactly 2 Bytes of data.
  if (Bytes.size() < 2) {
    Size = 0;
    return MCDisassembler::Fail;
  }

  if (!IsLittleEndian) {
    report_fatal_error("Big-endian mode currently is not supported!");
  } else {
    Insn = (Bytes[1] << 8) | Bytes[0];
  }

  return MCDisassembler::Success;
}

/// Read three bytes from the ArrayRef and return 24 bit data
static DecodeStatus readInstruction24(ArrayRef<uint8_t> Bytes, uint64_t Address,
                                      uint64_t &Size, uint64_t &Insn,
                                      bool IsLittleEndian) {
  // We want to read exactly 3 Bytes of data.
  if (Bytes.size() < 3) {
    Size = 0;
    return MCDisassembler::Fail;
  }

  if (!IsLittleEndian) {
    report_fatal_error("Big-endian mode currently is not supported!");
  } else {
    Insn = (Bytes[2] << 16) | (Bytes[1] << 8) | (Bytes[0] << 0);
  }

  return MCDisassembler::Success;
}

#include "XtensaGenDisassemblerTables.inc"

DecodeStatus XtensaDisassembler::getInstruction(MCInst &MI, uint64_t &Size,
                                                ArrayRef<uint8_t> Bytes,
                                                uint64_t Address,
                                                raw_ostream &CS) const {
  uint64_t Insn;
  DecodeStatus Result;

  // Parse 16-bit instructions
  if (hasDensity()) {
    Result = readInstruction16(Bytes, Address, Size, Insn, IsLittleEndian);
    if (Result == MCDisassembler::Fail)
      return MCDisassembler::Fail;
    LLVM_DEBUG(dbgs() << "Trying Xtensa 16-bit instruction table :\n");
    Result = decodeInstruction(DecoderTable16, MI, Insn, Address, this, STI);
    if (Result != MCDisassembler::Fail) {
      Size = 2;
      return Result;
    }
  }

  // Parse Core 24-bit instructions
  Result = readInstruction24(Bytes, Address, Size, Insn, IsLittleEndian);
  if (Result == MCDisassembler::Fail)
    return MCDisassembler::Fail;
  LLVM_DEBUG(dbgs() << "Trying Xtensa 24-bit instruction table :\n");
  Result = decodeInstruction(DecoderTable24, MI, Insn, Address, this, STI);
  if (Result != MCDisassembler::Fail) {
    Size = 3;
    return Result;
  }
  return Result;
}
