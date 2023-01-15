//===- LoongArchAsmPrinter.cpp - LoongArch LLVM Assembly Printer -*- C++ -*--=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format LoongArch assembly language.
//
//===----------------------------------------------------------------------===//

#include "LoongArchAsmPrinter.h"
#include "LoongArch.h"
#include "LoongArchTargetMachine.h"
#include "MCTargetDesc/LoongArchInstPrinter.h"
#include "TargetInfo/LoongArchTargetInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "loongarch-asm-printer"

// Simple pseudo-instructions have their lowering (with expansion to real
// instructions) auto-generated.
#include "LoongArchGenMCPseudoLowering.inc"

void LoongArchAsmPrinter::emitInstruction(const MachineInstr *MI) {
  LoongArch_MC::verifyInstructionPredicates(
      MI->getOpcode(), getSubtargetInfo().getFeatureBits());

  // Do any auto-generated pseudo lowerings.
  if (emitPseudoExpansionLowering(*OutStreamer, MI))
    return;

  MCInst TmpInst;
  if (!lowerLoongArchMachineInstrToMCInst(MI, TmpInst, *this))
    EmitToStreamer(*OutStreamer, TmpInst);
}

bool LoongArchAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                          const char *ExtraCode,
                                          raw_ostream &OS) {
  // First try the generic code, which knows about modifiers like 'c' and 'n'.
  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS))
    return false;

  const MachineOperand &MO = MI->getOperand(OpNo);
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0)
      return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      return true; // Unknown modifier.
    case 'z':      // Print $zero register if zero, regular printing otherwise.
      if (MO.isImm() && MO.getImm() == 0) {
        OS << '$' << LoongArchInstPrinter::getRegisterName(LoongArch::R0);
        return false;
      }
      break;
      // TODO: handle other extra codes if any.
    }
  }

  switch (MO.getType()) {
  case MachineOperand::MO_Immediate:
    OS << MO.getImm();
    return false;
  case MachineOperand::MO_Register:
    OS << '$' << LoongArchInstPrinter::getRegisterName(MO.getReg());
    return false;
  case MachineOperand::MO_GlobalAddress:
    PrintSymbolOperand(MO, OS);
    return false;
  default:
    llvm_unreachable("not implemented");
  }

  return true;
}

bool LoongArchAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                                unsigned OpNo,
                                                const char *ExtraCode,
                                                raw_ostream &OS) {
  // TODO: handle extra code.
  if (ExtraCode)
    return true;

  // We only support memory operands like "Base + Offset", where base must be a
  // register, and offset can be a register or an immediate value.
  const MachineOperand &BaseMO = MI->getOperand(OpNo);
  // Base address must be a register.
  if (!BaseMO.isReg())
    return true;
  // Print the base address register.
  OS << "$" << LoongArchInstPrinter::getRegisterName(BaseMO.getReg());
  // Print the offset operand.
  const MachineOperand &OffsetMO = MI->getOperand(OpNo + 1);
  if (OffsetMO.isReg())
    OS << ", $" << LoongArchInstPrinter::getRegisterName(OffsetMO.getReg());
  else if (OffsetMO.isImm())
    OS << ", " << OffsetMO.getImm();
  else
    return true;

  return false;
}

bool LoongArchAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  AsmPrinter::runOnMachineFunction(MF);
  return true;
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeLoongArchAsmPrinter() {
  RegisterAsmPrinter<LoongArchAsmPrinter> X(getTheLoongArch32Target());
  RegisterAsmPrinter<LoongArchAsmPrinter> Y(getTheLoongArch64Target());
}
