//=-- DPUMCInstLower.cpp - Convert DPU MachineInstr to an MCInst ------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower DPU MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "DPUMCInstLower.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <llvm/Support/Debug.h>

using namespace llvm;

#define DEBUG_TYPE "dpu-mclower"

MCSymbol *
DPUMCInstLower::GetGlobalAddressSymbol(const MachineOperand &MO) const {
  return Printer.getSymbol(MO.getGlobal());
}

MCSymbol *DPUMCInstLower::GetExternalSymbol(const MachineOperand &MO) const {
  return Printer.GetExternalSymbolSymbol(MO.getSymbolName());
}

MCOperand DPUMCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                             MCSymbol *Sym) const {

  const MCExpr *Expr = MCSymbolRefExpr::create(Sym, Ctx);
  LLVM_DEBUG(dbgs() << "DPU/Lower - Lowering symbol operand of type "
                    << MO.getType() << "\n");

  // Supported operands: register, immediate (32/64), Frame Index,
  // Constant Pool Index, Target Index=index+offset,
  // Jump Table Index, Global, External Symbol,
  if (MO.isReg() || MO.isImm() || MO.isCImm() || MO.isFI() || MO.isCPI() ||
      MO.isTargetIndex() || MO.isJTI() || MO.isGlobal() || MO.isSymbol()) {
    return MCOperand::createExpr(Expr);
  } else if (MO.isFPImm()) {
    // FP not supported
    report_fatal_error("FP symbol operand lowering not supported", true);
  } else if (MO.isMBB()) {
    report_fatal_error("MBB symbol operand lowering not supported", true);
  } else if (MO.isBlockAddress()) {
    report_fatal_error("Block Address symbol operand lowering not supported",
                       true);
  } else if (MO.isRegMask()) {
    report_fatal_error("Register mask symbol operand lowering not supported",
                       true);
  } else {
    // All other types...
    report_fatal_error("Unsupported symbol operand type", true);
  }
}

MCSymbol *DPUMCInstLower::GetJumpTableSymbol(const MachineOperand &MO) const {
  return Printer.GetJTISymbol(MO.getIndex());
}

void DPUMCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  LLVM_DEBUG({
    dbgs() << "DPU/Lower - Lowering machine operands for ";
    MI->dump();
  });
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);

    MCOperand MCOp;
    switch (MO.getType()) {
    case MachineOperand::MO_Register:
      // Ignore all implicit register operands.
      if (MO.isImplicit())
        continue;
      MCOp = MCOperand::createReg(MO.getReg());
      break;

    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::createImm(MO.getImm());
      break;

    case MachineOperand::MO_MachineBasicBlock:
      MCOp = MCOperand::createExpr(
          MCSymbolRefExpr::create(MO.getMBB()->getSymbol(), Ctx));
      break;

    case MachineOperand::MO_RegisterMask:
      continue;

    case MachineOperand::MO_GlobalAddress:
      MCOp = LowerSymbolOperand(MO, GetGlobalAddressSymbol(MO));
      break;

    case MachineOperand::MO_JumpTableIndex:
      // The index is an immediate value, more specifically a symbol. We can
      // safely
      // lower it as an immediate.
      // The function will generate something like "LJTIX_X", which is what the
      // rest of the code will refer to (some kind of magic, as usual).
      MCOp = LowerSymbolOperand(MO, GetJumpTableSymbol(MO));
      break;

    case MachineOperand::MO_ExternalSymbol:
      MCOp = LowerSymbolOperand(MO, GetExternalSymbol(MO));
      break;

    default:
      llvm_unreachable("DPU/Lower - unknown operand type");
    }

    OutMI.addOperand(MCOp);
  }
}
