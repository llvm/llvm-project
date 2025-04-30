//=-- Next32MCInstLower.cpp - Convert Next32 MachineInstr to an MCInst ------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower Next32 MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "Next32MCInstLower.h"
#include "MCTargetDesc/Next32MCExpr.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

MCOperand
Next32MCInstLower::LowerGlobalAddress(const MachineOperand &MO) const {
  MCSymbol *Sym = Printer.getSymbol(MO.getGlobal());
  const MCSymbolRefExpr *SymExpr = MCSymbolRefExpr::create(Sym, Ctx);
  const MCExpr *Next32Expr = Next32MCExpr::create(
      (Next32MCExpr::Next32ExprKind)MO.getTargetFlags(), SymExpr, Ctx);
  return MCOperand::createExpr(Next32Expr);
}

MCOperand Next32MCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                                MCSymbol *Sym) const {

  const MCExpr *Expr = MCSymbolRefExpr::create(Sym, Ctx);

  if (!MO.isJTI() && MO.getOffset())
    llvm_unreachable("unknown symbol op");

  return MCOperand::createExpr(Expr);
}

void Next32MCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    MCOperand MCOp;
    switch (MO.getType()) {
    default:
      MI->print(errs());
      llvm_unreachable("unknown operand type");
    case MachineOperand::MO_Register:
      // Ignore all implicit register operands.
      if (MO.isImplicit())
        continue;
      MCOp = MCOperand::createReg(MO.getReg());
      break;
    case MachineOperand::MO_Immediate: {
      MCOp = MCOperand::createImm(MO.getImm());
      break;
    }
    case MachineOperand::MO_MachineBasicBlock:
      MCOp = MCOperand::createExpr(
          MCSymbolRefExpr::create(MO.getMBB()->getSymbol(), Ctx));
      break;
    case MachineOperand::MO_RegisterMask:
      continue;
    case MachineOperand::MO_MCSymbol:
      MCOp = LowerSymbolOperand(MO, MO.getMCSymbol());
      break;
    case MachineOperand::MO_GlobalAddress:
      MCOp = LowerGlobalAddress(MO);
      break;
    }

    OutMI.addOperand(MCOp);
  }
}
