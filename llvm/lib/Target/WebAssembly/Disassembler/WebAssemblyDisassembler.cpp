//==- WebAssemblyDisassembler.cpp - Disassembler for WebAssembly -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file is part of the WebAssembly Disassembler.
///
/// It contains code to translate the data produced by the decoder into
/// MCInsts.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-disassembler"

namespace {
class WebAssemblyDisassembler final : public MCDisassembler {
  std::unique_ptr<const MCInstrInfo> MCII;

  DecodeStatus getInstruction(MCInst &Instr, uint64_t &Size,
                              ArrayRef<uint8_t> Bytes, uint64_t Address,
                              raw_ostream &VStream,
                              raw_ostream &CStream) const override;

public:
  WebAssemblyDisassembler(const MCSubtargetInfo &STI, MCContext &Ctx,
                          std::unique_ptr<const MCInstrInfo> MCII)
      : MCDisassembler(STI, Ctx), MCII(std::move(MCII)) {}
};
} // end anonymous namespace

static MCDisassembler *createWebAssemblyDisassembler(const Target &T,
                                                     const MCSubtargetInfo &STI,
                                                     MCContext &Ctx) {
  std::unique_ptr<const MCInstrInfo> MCII(T.createMCInstrInfo());
  return new WebAssemblyDisassembler(STI, Ctx, std::move(MCII));
}

extern "C" void LLVMInitializeWebAssemblyDisassembler() {
  // Register the disassembler for each target.
  TargetRegistry::RegisterMCDisassembler(TheWebAssemblyTarget32,
                                         createWebAssemblyDisassembler);
  TargetRegistry::RegisterMCDisassembler(TheWebAssemblyTarget64,
                                         createWebAssemblyDisassembler);
}

MCDisassembler::DecodeStatus WebAssemblyDisassembler::getInstruction(
    MCInst &MI, uint64_t &Size, ArrayRef<uint8_t> Bytes, uint64_t /*Address*/,
    raw_ostream &OS, raw_ostream &CS) const {
  Size = 0;
  uint64_t Pos = 0;

  // Read the opcode.
  if (Pos + sizeof(uint64_t) > Bytes.size())
    return MCDisassembler::Fail;
  uint64_t Opcode = support::endian::read64le(Bytes.data() + Pos);
  Pos += sizeof(uint64_t);

  if (Opcode >= WebAssembly::INSTRUCTION_LIST_END)
    return MCDisassembler::Fail;

  MI.setOpcode(Opcode);
  const MCInstrDesc &Desc = MCII->get(Opcode);
  unsigned NumFixedOperands = Desc.NumOperands;

  // If it's variadic, read the number of extra operands.
  unsigned NumExtraOperands = 0;
  if (Desc.isVariadic()) {
    if (Pos + sizeof(uint64_t) > Bytes.size())
      return MCDisassembler::Fail;
    NumExtraOperands = support::endian::read64le(Bytes.data() + Pos);
    Pos += sizeof(uint64_t);
  }

  // Read the fixed operands. These are described by the MCInstrDesc.
  for (unsigned i = 0; i < NumFixedOperands; ++i) {
    const MCOperandInfo &Info = Desc.OpInfo[i];
    switch (Info.OperandType) {
    case MCOI::OPERAND_IMMEDIATE:
    case WebAssembly::OPERAND_BASIC_BLOCK: {
      if (Pos + sizeof(uint64_t) > Bytes.size())
        return MCDisassembler::Fail;
      uint64_t Imm = support::endian::read64le(Bytes.data() + Pos);
      Pos += sizeof(uint64_t);
      MI.addOperand(MCOperand::createImm(Imm));
      break;
    }
    case MCOI::OPERAND_REGISTER: {
      if (Pos + sizeof(uint64_t) > Bytes.size())
        return MCDisassembler::Fail;
      uint64_t Reg = support::endian::read64le(Bytes.data() + Pos);
      Pos += sizeof(uint64_t);
      MI.addOperand(MCOperand::createReg(Reg));
      break;
    }
    case WebAssembly::OPERAND_FPIMM: {
      // TODO: MC converts all floating point immediate operands to double.
      // This is fine for numeric values, but may cause NaNs to change bits.
      if (Pos + sizeof(uint64_t) > Bytes.size())
        return MCDisassembler::Fail;
      uint64_t Bits = support::endian::read64le(Bytes.data() + Pos);
      Pos += sizeof(uint64_t);
      double Imm;
      memcpy(&Imm, &Bits, sizeof(Imm));
      MI.addOperand(MCOperand::createFPImm(Imm));
      break;
    }
    default:
      llvm_unreachable("unimplemented operand kind");
    }
  }

  // Read the extra operands.
  assert(NumExtraOperands == 0 || Desc.isVariadic());
  for (unsigned i = 0; i < NumExtraOperands; ++i) {
    if (Pos + sizeof(uint64_t) > Bytes.size())
      return MCDisassembler::Fail;
    if (Desc.TSFlags & WebAssemblyII::VariableOpIsImmediate) {
      // Decode extra immediate operands.
      uint64_t Imm = support::endian::read64le(Bytes.data() + Pos);
      MI.addOperand(MCOperand::createImm(Imm));
    } else {
      // Decode extra register operands.
      uint64_t Reg = support::endian::read64le(Bytes.data() + Pos);
      MI.addOperand(MCOperand::createReg(Reg));
    }
    Pos += sizeof(uint64_t);
  }

  Size = Pos;
  return MCDisassembler::Success;
}
