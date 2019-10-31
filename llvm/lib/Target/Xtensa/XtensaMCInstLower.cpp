//===- XtensaMCInstLower.cpp - Convert Xtensa MachineInstr to MCInst ------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower Xtensa MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "XtensaMCInstLower.h"
#include "MCTargetDesc/XtensaMCExpr.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

XtensaMCInstLower::XtensaMCInstLower(MCContext &ctx,
                                     XtensaAsmPrinter &asmPrinter)
    : Ctx(ctx), Printer(asmPrinter) {}

MCSymbol *
XtensaMCInstLower::GetConstantPoolIndexSymbol(const MachineOperand &MO) const {
  // Create a symbol for the name.
  return Printer.GetCPISymbol(MO.getIndex());
}

MCOperand
XtensaMCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                      MachineOperand::MachineOperandType MOTy,
                                      unsigned Offset) const {
  const MCSymbol *Symbol;
  XtensaMCExpr::VariantKind Kind = XtensaMCExpr::VK_Xtensa_None;

  switch (MOTy) {
  case MachineOperand::MO_MachineBasicBlock:
    Symbol = MO.getMBB()->getSymbol();
    break;
  case MachineOperand::MO_GlobalAddress:
    Symbol = Printer.getSymbol(MO.getGlobal());
    Offset += MO.getOffset();
    break;
  case MachineOperand::MO_BlockAddress:
    Symbol = Printer.GetBlockAddressSymbol(MO.getBlockAddress());
    Offset += MO.getOffset();
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    Symbol = GetConstantPoolIndexSymbol(MO);
    Offset += MO.getOffset();
    break;
  default:
    llvm_unreachable("<unknown operand type>");
  }

  const MCExpr *ME =
      MCSymbolRefExpr::create(Symbol, MCSymbolRefExpr::VK_None, Ctx);

  ME = XtensaMCExpr::create(ME, Kind, Ctx);

  if (Offset) {
    // Assume offset is never negative.
    assert(Offset > 0);

    const MCConstantExpr *OffsetExpr = MCConstantExpr::create(Offset, Ctx);
    ME = MCBinaryExpr::createAdd(ME, OffsetExpr, Ctx);
  }

  return MCOperand::createExpr(ME);
}

MCOperand XtensaMCInstLower::lowerOperand(const MachineOperand &MO,
                                          unsigned Offset) const {
  MachineOperand::MachineOperandType MOTy = MO.getType();

  switch (MOTy) {
  case MachineOperand::MO_Register:
    // Ignore all implicit register operands.
    if (MO.isImplicit())
      break;
    return MCOperand::createReg(MO.getReg());
  case MachineOperand::MO_Immediate:
    return MCOperand::createImm(MO.getImm() + Offset);
  case MachineOperand::MO_RegisterMask:
    break;
  case MachineOperand::MO_ConstantPoolIndex:
    return LowerSymbolOperand(MO, MOTy, Offset);
  default:
    llvm_unreachable("unknown operand type");
  }

  return MCOperand();
}

void XtensaMCInstLower::lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    MCOperand MCOp = lowerOperand(MO);

    if (MCOp.isValid())
      OutMI.addOperand(MCOp);
  }
}
