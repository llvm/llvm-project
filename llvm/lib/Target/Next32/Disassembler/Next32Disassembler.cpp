//===-- Next32Disassembler.cpp - Disassembler for Next32 ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the Next32 Disassembler.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/Next32MCTargetDesc.h"
#include "Next32.h"
#include "Next32Subtarget.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDecoderOps.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "next32-disassembler"

typedef MCDisassembler::DecodeStatus DecodeStatus;

namespace {

/// A disassembler class for Next32.
class Next32Disassembler : public MCDisassembler {
  std::unique_ptr<const MCInstrInfo> MII;

public:
  Next32Disassembler(const MCSubtargetInfo &STI, MCContext &Ctx,
                     std::unique_ptr<const MCInstrInfo> MII);
  ~Next32Disassembler() override = default;

  DecodeStatus getInstruction(MCInst &Instr, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &CStream) const override;
};

} // end anonymous namespace

Next32Disassembler::Next32Disassembler(const MCSubtargetInfo &STI,
                                       MCContext &Ctx,
                                       std::unique_ptr<const MCInstrInfo> MII)
    : MCDisassembler(STI, Ctx), MII(std::move(MII)) {}

static MCDisassembler *createNext32Disassembler(const Target &T,
                                                const MCSubtargetInfo &STI,
                                                MCContext &Ctx) {
  std::unique_ptr<const MCInstrInfo> MII(T.createMCInstrInfo());
  return new Next32Disassembler(STI, Ctx, std::move(MII));
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeNext32Disassembler() {
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(getTheNext32Target(),
                                         createNext32Disassembler);
}

namespace llvm {
extern const uint16_t Next32RegEncodingTable[];
}

static DecodeStatus DecodeGPR(MCInst &Inst, unsigned RegNo) {
  for (size_t i = 0; i < Next32::NUM_TARGET_REGS; ++i)
    if (Next32RegEncodingTable[i] == RegNo) {
      Inst.addOperand(MCOperand::createReg(i));
      return MCDisassembler::Success;
    }

  return MCDisassembler::Fail;
}

static DecodeStatus DecodeGPR32RegisterClass(MCInst &Inst, unsigned RegNo,
                                             uint64_t, const void *) {
  return DecodeGPR(Inst, RegNo);
}

static DecodeStatus DecodeDUPInstruction(MCInst &Inst, uint64_t Insn,
                                         uint64_t Address, const void *Decoder);

#include "Next32GenDisassemblerTables.inc"

static DecodeStatus DecodeDUPInstruction(MCInst &Inst, uint64_t Insn,
                                         uint64_t Address,
                                         const void *Decoder) {
  uint64_t tmp = fieldFromInstruction(Insn, 0, 16);
  if (DecodeGPR32RegisterClass(Inst, tmp, Address, Decoder) ==
      MCDisassembler::Fail) {
    return MCDisassembler::Fail;
  }
  if (DecodeGPR32RegisterClass(Inst, tmp, Address, Decoder) ==
      MCDisassembler::Fail) {
    return MCDisassembler::Fail;
  }
  tmp = fieldFromInstruction(Insn, 16, 16);
  if (DecodeGPR32RegisterClass(Inst, tmp, Address, Decoder) ==
      MCDisassembler::Fail) {
    return MCDisassembler::Fail;
  }
  tmp = fieldFromInstruction(Insn, 60, 4);
  Inst.addOperand(MCOperand::createImm(tmp));
  tmp = fieldFromInstruction(Insn, 59, 1);
  Inst.addOperand(MCOperand::createImm(tmp));
  tmp = fieldFromInstruction(Insn, 32, 16);
  Inst.addOperand(MCOperand::createImm(tmp));
  return MCDisassembler::Success;
}

DecodeStatus Next32Disassembler::getInstruction(MCInst &Instr, uint64_t &Size,
                                                ArrayRef<uint8_t> Bytes,
                                                uint64_t Address,
                                                raw_ostream &CStream) const {
  const DecodeStatus Result = decodeInstruction(
      DecoderTableNext3264, Instr, support::endian::read64be(&Bytes[0]),
      Address, this, STI);

  if (Result == MCDisassembler::Fail)
    return MCDisassembler::Fail;

  const MCInstrDesc &Desc = MII->get(Instr.getOpcode());
  if (!(Desc.TSFlags & Next32II::Is128BitRRRRInstruction)) {
    Size = 8;
    return MCDisassembler::Success;
  }

  Size = 16;
  const uint64_t InstructionSecondPart = support::endian::read64be(&Bytes[8]);
  const uint64_t Src1 = fieldFromInstruction(InstructionSecondPart, 48, 16);
  const uint64_t Src2 = fieldFromInstruction(InstructionSecondPart, 32, 16);
  const uint64_t AddrSpace = fieldFromInstruction(InstructionSecondPart, 3, 3);
  const uint64_t Count = fieldFromInstruction(InstructionSecondPart, 0, 3);

  // This code assumes that SrcReg1 and SrcReg2 are always last in the MCInst
  // operand list (because DecodeGPR32RegisterClass appends to the list). This
  // doesn't have to be the case because the MCInst operand list starts with the
  // instruction's 'outs' and ends with the instruction's 'ins', where inout
  // registers appear twice (in and out are constrained to be the same
  // register). But decodeInstruction above only has visibility into the fields
  // of the first part of the instruction, So this code it will not properly
  // handle cases where the operand list post-decodeInstruction already contains
  // both ins and outs. For example, because the first part contains CondReg,
  // DstReg1 and DstReg2, these will not order the operands properly:
  // 1. Usage of the condition register (pure-in) when DstReg1/2 are pure-out
  // 2. Either of DstReg1/2 being in-out rather than both being pure-in or both
  // being pure-out

  if ((Desc.TSFlags & Next32II::Is128BitRRRRInstructionSrcReg1Out) &&
      DecodeGPR32RegisterClass(Instr, Src1, Address, this) ==
          MCDisassembler::Fail)
    return MCDisassembler::Fail;

  if ((Desc.TSFlags & Next32II::Is128BitRRRRInstructionSrcReg2Out) &&
      DecodeGPR32RegisterClass(Instr, Src2, Address, this) ==
          MCDisassembler::Fail)
    return MCDisassembler::Fail;

  if ((Desc.TSFlags & Next32II::Is128BitRRRRInstructionSrcReg1In) &&
      DecodeGPR32RegisterClass(Instr, Src1, Address, this) ==
          MCDisassembler::Fail)
    return MCDisassembler::Fail;

  if ((Desc.TSFlags & Next32II::Is128BitRRRRInstructionSrcReg2In) &&
      DecodeGPR32RegisterClass(Instr, Src2, Address, this) ==
          MCDisassembler::Fail)
    return MCDisassembler::Fail;

  if (Desc.TSFlags & Next32II::is128BitRRRRInstructionWithCount)
    Instr.addOperand(MCOperand::createImm(Count));

  if (Desc.TSFlags & Next32II::is128BitRRRRInstructionWithAddrSpace)
    Instr.addOperand(MCOperand::createImm(AddrSpace));

  return MCDisassembler::Success;
}
