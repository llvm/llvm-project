//===-- SuperHMCInstLower.cpp - Lower MachineInstr to MCInst ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SuperHMCInstLower.h"
#include "MCTargetDesc/SuperHMCAsmInfo.h"
#include "MCTargetDesc/SuperHBaseInfo.h"
#include "SuperHSubtarget.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"


namespace llvm {

MCOperand
SuperHMCInstLower::lowerSymbolOperand(const MachineOperand &MO, MCSymbol *Sym,
                                      const SuperHSubtarget &Subtarget) const {
  const MCExpr *Expr = nullptr;
  switch (MO.getTargetFlags()) {
    case SHII::MO_NO_FLAG:
      Expr = MCSymbolRefExpr::create(Sym, SH::S_None, Ctx);
      break;

    case SHII::MO_GOT:
      Expr = MCSymbolRefExpr::create(Sym, SH::S_GOT, Ctx);
      break;

    case SHII::MO_GOTPC:
      Expr = MCSymbolRefExpr::create(Sym, SH::S_GOT_PCREL, Ctx);
      break;

    case SHII::MO_GOTOFF:
      Expr = MCSymbolRefExpr::create(Sym, SH::S_GOT_OFF, Ctx);
      break;

    case SHII::MO_DIR:
      Expr = MCSymbolRefExpr::create(Sym, SH::S_DIR, Ctx);
      break;

    case SHII::MO_PCREL:
      Expr = MCSymbolRefExpr::create(Sym, SH::S_PCREL, Ctx);
      break;
  }
  return MCOperand::createExpr(Expr);
}

void SuperHMCInstLower::lowerInstruction(const MachineInstr &MI,
                                         MCInst &OutMI) const {
  auto &Subtarget = MI.getParent()->getParent()->getSubtarget<SuperHSubtarget>();

  OutMI.setOpcode(MI.getOpcode());
  for (MachineOperand const &MO : MI.operands()) {
    MCOperand MCOp;

    switch (MO.getType()) {
    default:
      MI.print(errs());
      llvm_unreachable("unknown operand type");
    case MachineOperand::MO_Register:
      // Ignore all implicit register operands.
      if (MO.isImplicit())
        continue;
      MCOp = MCOperand::createReg(MO.getReg());
      break;
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::createImm(MO.getImm());
      break;
    case MachineOperand::MO_GlobalAddress:
      MCOp =
          lowerSymbolOperand(MO, Printer.getSymbol(MO.getGlobal()), Subtarget);
      break;
    case MachineOperand::MO_ExternalSymbol:
      MCOp = lowerSymbolOperand(
          MO, Printer.GetExternalSymbolSymbol(MO.getSymbolName()), Subtarget);
      break;
    case MachineOperand::MO_MachineBasicBlock:

      // NOTE:  Branch instructions will generally jump to labels.
      //        so ensure we emit them if they're referenced in an
      //        operand during lowering.
      MO.getMBB()->setLabelMustBeEmitted();
      MCOp = MCOperand::createExpr(
          MCSymbolRefExpr::create(MO.getMBB()->getSymbol(), Ctx));
      break;
    case MachineOperand::MO_RegisterMask:
      continue;
    case MachineOperand::MO_BlockAddress:
      MCOp = lowerSymbolOperand(
          MO, Printer.GetBlockAddressSymbol(MO.getBlockAddress()), Subtarget);
      break;
    case MachineOperand::MO_JumpTableIndex:
      MCOp = lowerSymbolOperand(MO, Printer.GetJTISymbol(MO.getIndex()),
                                Subtarget);
      break;
    case MachineOperand::MO_ConstantPoolIndex:
      MCOp = lowerSymbolOperand(MO, Printer.GetCPISymbol(MO.getIndex()),
                                Subtarget);
      break;
    }

    OutMI.addOperand(MCOp);
  }
}

} // namespace llvm