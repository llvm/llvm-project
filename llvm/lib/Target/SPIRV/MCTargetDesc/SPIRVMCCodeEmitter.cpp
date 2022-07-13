//===-- SPIRVMCCodeEmitter.cpp - Emit SPIR-V machine code -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPIRVMCCodeEmitter class.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SPIRVMCTargetDesc.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;

#define DEBUG_TYPE "spirv-mccodeemitter"

namespace {

class SPIRVMCCodeEmitter : public MCCodeEmitter {
  const MCInstrInfo &MCII;

public:
  SPIRVMCCodeEmitter(const MCInstrInfo &mcii) : MCII(mcii) {}
  SPIRVMCCodeEmitter(const SPIRVMCCodeEmitter &) = delete;
  void operator=(const SPIRVMCCodeEmitter &) = delete;
  ~SPIRVMCCodeEmitter() override = default;

  // getBinaryCodeForInstr - TableGen'erated function for getting the
  // binary encoding for an instruction.
  uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const;

  void encodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;
};

} // end anonymous namespace

MCCodeEmitter *llvm::createSPIRVMCCodeEmitter(const MCInstrInfo &MCII,
                                              MCContext &Ctx) {
  return new SPIRVMCCodeEmitter(MCII);
}

using EndianWriter = support::endian::Writer;

// Check if the instruction has a type argument for operand 1, and defines an ID
// output register in operand 0. If so, we need to swap operands 0 and 1 so the
// type comes first in the output, despide coming second in the MCInst.
static bool hasType(const MCInst &MI, const MCInstrInfo &MII) {
  MCInstrDesc MCDesc = MII.get(MI.getOpcode());
  // If we define an output, and have at least one other argument.
  if (MCDesc.getNumDefs() == 1 && MCDesc.getNumOperands() >= 2) {
    // Check if we define an ID, and take a type as operand 1.
    auto DefOpInfo = MCDesc.opInfo_begin();
    auto FirstArgOpInfo = MCDesc.opInfo_begin() + 1;
    return (DefOpInfo->RegClass == SPIRV::IDRegClassID ||
            DefOpInfo->RegClass == SPIRV::ANYIDRegClassID) &&
           FirstArgOpInfo->RegClass == SPIRV::TYPERegClassID;
  }
  return false;
}

static void emitOperand(const MCOperand &Op, EndianWriter &OSE) {
  if (Op.isReg()) {
    // Emit the id index starting at 1 (0 is an invalid index).
    OSE.write<uint32_t>(Register::virtReg2Index(Op.getReg()) + 1);
  } else if (Op.isImm()) {
    OSE.write<uint32_t>(Op.getImm());
  } else {
    llvm_unreachable("Unexpected operand type in VReg");
  }
}

// Emit the type in operand 1 before the ID in operand 0 it defines, and all
// remaining operands in the order they come naturally.
static void emitTypedInstrOperands(const MCInst &MI, EndianWriter &OSE) {
  unsigned NumOps = MI.getNumOperands();
  emitOperand(MI.getOperand(1), OSE);
  emitOperand(MI.getOperand(0), OSE);
  for (unsigned i = 2; i < NumOps; ++i)
    emitOperand(MI.getOperand(i), OSE);
}

// Emit operands in the order they come naturally.
static void emitUntypedInstrOperands(const MCInst &MI, EndianWriter &OSE) {
  for (const auto &Op : MI)
    emitOperand(Op, OSE);
}

void SPIRVMCCodeEmitter::encodeInstruction(const MCInst &MI, raw_ostream &OS,
                                           SmallVectorImpl<MCFixup> &Fixups,
                                           const MCSubtargetInfo &STI) const {
  EndianWriter OSE(OS, support::little);

  // Encode the first 32 SPIR-V bytes with the number of args and the opcode.
  const uint64_t OpCode = getBinaryCodeForInstr(MI, Fixups, STI);
  const uint32_t NumWords = MI.getNumOperands() + 1;
  const uint32_t FirstWord = (NumWords << 16) | OpCode;
  OSE.write<uint32_t>(FirstWord);

  // Emit the instruction arguments (emitting the output type first if present).
  if (hasType(MI, MCII))
    emitTypedInstrOperands(MI, OSE);
  else
    emitUntypedInstrOperands(MI, OSE);
}

#include "SPIRVGenMCCodeEmitter.inc"
