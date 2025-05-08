//===- ParasolDisassembler.cpp - Disassembler for Parasol -----------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file is part of the Parasol Disassembler.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ParasolMCTargetDesc.h"
#include "TargetInfo/ParasolTargetInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDecoderOps.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "parasol-disassembler"

typedef MCDisassembler::DecodeStatus DecodeStatus;

namespace {

/// A disassembler class for Parasol.
class ParasolDisassembler : public MCDisassembler {
public:
  ParasolDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx)
      : MCDisassembler(STI, Ctx) {}
  virtual ~ParasolDisassembler() = default;

  DecodeStatus getInstruction(MCInst &Instr, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &CStream) const override;
};
} // namespace

static MCDisassembler *createParasolDisassembler(const Target &T,
                                                 const MCSubtargetInfo &STI,
                                                 MCContext &Ctx) {
  return new ParasolDisassembler(STI, Ctx);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeParasolDisassembler() {
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(getTheParasolTarget(),
                                         createParasolDisassembler);
}

static const unsigned IntRegDecoderTable[] = {
    Parasol::R0,  Parasol::R1,  Parasol::R2,  Parasol::R3,  Parasol::R4,
    Parasol::R5,  Parasol::R6,  Parasol::R7,  Parasol::R8,  Parasol::R9,
    Parasol::R10, Parasol::R11, Parasol::R12, Parasol::R13, Parasol::R14,
    Parasol::R15, Parasol::R16, Parasol::R17, Parasol::R18, Parasol::R19,
    Parasol::R20, Parasol::R21, Parasol::R22, Parasol::R23, Parasol::R24,
    Parasol::R25, Parasol::R26, Parasol::R27, Parasol::R28, Parasol::R29,
    Parasol::R30, Parasol::R31, Parasol::R32, Parasol::R33, Parasol::R34,
    Parasol::R35, Parasol::R36, Parasol::R37, Parasol::R38, Parasol::R39,
    Parasol::R40, Parasol::R41, Parasol::R42, Parasol::R43, Parasol::R44,
    Parasol::R45, Parasol::R46, Parasol::R47, Parasol::R48, Parasol::R49,
    Parasol::R50, Parasol::R51, Parasol::R52, Parasol::R53, Parasol::R54,
    Parasol::R55, Parasol::R56, Parasol::R57, Parasol::R58, Parasol::R59,
    Parasol::R60, Parasol::R61, Parasol::R62, Parasol::R63,
};

static const unsigned PointerRegDecoderTable[] = {
    Parasol::P0,  Parasol::P1,  Parasol::P2,  Parasol::P3,  Parasol::P4,
    Parasol::P5,  Parasol::P6,  Parasol::P7,  Parasol::P8,  Parasol::P9,
    Parasol::P10, Parasol::P11, Parasol::P12, Parasol::P13, Parasol::P14,
    Parasol::P15, Parasol::P16, Parasol::P17, Parasol::P18, Parasol::P19,
    Parasol::P20, Parasol::P21, Parasol::P22, Parasol::P23, Parasol::P24,
    Parasol::P25, Parasol::P26, Parasol::P27, Parasol::P28, Parasol::P29,
    Parasol::P30, Parasol::P31, Parasol::P32, Parasol::P33, Parasol::P34,
    Parasol::P35, Parasol::P36, Parasol::P37, Parasol::P38, Parasol::P39,
    Parasol::P40, Parasol::P41, Parasol::P42, Parasol::P43, Parasol::P44,
    Parasol::P45, Parasol::P46, Parasol::P47, Parasol::P48, Parasol::P49,
    Parasol::P50, Parasol::P51, Parasol::P52, Parasol::P53, Parasol::P54,
    Parasol::P55, Parasol::P56, Parasol::P57, Parasol::P58, Parasol::P59,
    Parasol::P60, Parasol::P61, Parasol::P62, Parasol::P63,
};

static const unsigned GeneralRegDecoderTable[] = {
    Parasol::X0,  Parasol::X1,  Parasol::X2,  Parasol::X3,  Parasol::X4,
    Parasol::X5,  Parasol::X6,  Parasol::X7,  Parasol::X8,  Parasol::X9,
    Parasol::X10, Parasol::X11, Parasol::X12, Parasol::X13, Parasol::X14,
    Parasol::X15, Parasol::X16, Parasol::X17, Parasol::X18, Parasol::X19,
    Parasol::X20, Parasol::X21, Parasol::X22, Parasol::X23, Parasol::X24,
    Parasol::X25, Parasol::X26, Parasol::X27, Parasol::X28, Parasol::X29,
    Parasol::X30, Parasol::X31,
};

static DecodeStatus DecodeIRRegisterClass(MCInst &Inst, unsigned RegNo,
                                          uint64_t Address,
                                          const MCDisassembler *Decoder) {
  if (RegNo > 63)
    return MCDisassembler::Fail;
  unsigned Reg = IntRegDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodePRRegisterClass(MCInst &Inst, unsigned RegNo,
                                          uint64_t Address,
                                          const MCDisassembler *Decoder) {
  if (RegNo > 63)
    return MCDisassembler::Fail;
  unsigned Reg = PointerRegDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

static DecodeStatus DecodeGPRRegisterClass(MCInst &Inst, unsigned RegNo,
                                           uint64_t Address,
                                           const MCDisassembler *Decoder) {
  if (RegNo > 31)
    return MCDisassembler::Fail;
  unsigned Reg = GeneralRegDecoderTable[RegNo];
  Inst.addOperand(MCOperand::createReg(Reg));
  return MCDisassembler::Success;
}

#include "ParasolGenDisassemblerTables.inc"

/// Read eight bytes from the ArrayRef and return 64 bit word.
static DecodeStatus readInstruction64(ArrayRef<uint8_t> Bytes, uint64_t Address,
                                      uint64_t &Size, uint64_t &Insn,
                                      bool IsLittleEndian) {
  // We want to read exactly 8 Bytes of data.
  if (Bytes.size() < 8) {
    Size = 0;
    return MCDisassembler::Fail;
  }

  Insn = ((uint64_t)Bytes[0] << 0) | ((uint64_t)Bytes[1] << 8) |
         ((uint64_t)Bytes[2] << 16) | ((uint64_t)Bytes[3] << 24) |
         ((uint64_t)Bytes[4] << 32) | ((uint64_t)Bytes[5] << 40) |
         ((uint64_t)Bytes[6] << 48) | ((uint64_t)Bytes[7] << 56);

  return MCDisassembler::Success;
}

DecodeStatus ParasolDisassembler::getInstruction(MCInst &Instr, uint64_t &Size,
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
  Result = decodeInstruction(DecoderTable64, Instr, Insn, Address, this, STI);

  if (Result != MCDisassembler::Fail) {
    Size = 8;
    return Result;
  }

  return MCDisassembler::Fail;
}
