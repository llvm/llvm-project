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
    Parasol::X0,  Parasol::X1,  Parasol::X2,  Parasol::X3,  Parasol::X4,
    Parasol::X5,  Parasol::X6,  Parasol::X7,  Parasol::X8,  Parasol::X9,
    Parasol::X10, Parasol::X11, Parasol::X12, Parasol::X13, Parasol::X14,
    Parasol::X15, Parasol::X16, Parasol::X17, Parasol::X18, Parasol::X19,
    Parasol::X20, Parasol::X21, Parasol::X22, Parasol::X23, Parasol::X24,
    Parasol::X25, Parasol::X26, Parasol::X27, Parasol::X28, Parasol::X29,
    Parasol::X30, Parasol::X31, Parasol::X32, Parasol::X33, Parasol::X34,
    Parasol::X35, Parasol::X36, Parasol::X37, Parasol::X38, Parasol::X39,
    Parasol::X40, Parasol::X41, Parasol::X42, Parasol::X43, Parasol::X44,
    Parasol::X45, Parasol::X46, Parasol::X47, Parasol::X48, Parasol::X49,
    Parasol::X50, Parasol::X51, Parasol::X52, Parasol::X53, Parasol::X54,
    Parasol::X55, Parasol::X56, Parasol::X57, Parasol::X58, Parasol::X59,
    Parasol::X60, Parasol::X61, Parasol::X62, Parasol::X63,
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
