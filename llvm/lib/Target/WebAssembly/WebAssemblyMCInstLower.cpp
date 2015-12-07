// WebAssemblyMCInstLower.cpp - Convert WebAssembly MachineInstr to an MCInst //
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains code to lower WebAssembly MachineInstrs to their
/// corresponding MCInst records.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyMCInstLower.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Constants.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

MCSymbol *
WebAssemblyMCInstLower::GetGlobalAddressSymbol(const MachineOperand &MO) const {
  return Printer.getSymbol(MO.getGlobal());
}

MCSymbol *WebAssemblyMCInstLower::GetExternalSymbolSymbol(
    const MachineOperand &MO) const {
  return Printer.GetExternalSymbolSymbol(MO.getSymbolName());
}

MCOperand WebAssemblyMCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                                     MCSymbol *Sym) const {
  assert(MO.getTargetFlags() == 0 && "WebAssembly does not use target flags");

  const MCExpr *Expr = MCSymbolRefExpr::create(Sym, Ctx);

  int64_t Offset = MO.getOffset();
  if (Offset != 0) {
    assert(!MO.isJTI() && "Unexpected offset with jump table index");
    Expr =
        MCBinaryExpr::createAdd(Expr, MCConstantExpr::create(Offset, Ctx), Ctx);
  }

  return MCOperand::createExpr(Expr);
}

void WebAssemblyMCInstLower::Lower(const MachineInstr *MI,
                                   MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);

    MCOperand MCOp;
    switch (MO.getType()) {
    default:
      MI->dump();
      llvm_unreachable("unknown operand type");
    case MachineOperand::MO_Register: {
      // Ignore all implicit register operands.
      if (MO.isImplicit())
        continue;
      const WebAssemblyFunctionInfo &MFI =
          *MI->getParent()->getParent()->getInfo<WebAssemblyFunctionInfo>();
      unsigned WAReg = MFI.getWAReg(MO.getReg());
      MCOp = MCOperand::createReg(WAReg);
      break;
    }
    case MachineOperand::MO_Immediate:
      MCOp = MCOperand::createImm(MO.getImm());
      break;
    case MachineOperand::MO_FPImmediate: {
      // TODO: MC converts all floating point immediate operands to double.
      // This is fine for numeric values, but may cause NaNs to change bits.
      const ConstantFP *Imm = MO.getFPImm();
      if (Imm->getType()->isFloatTy())
        MCOp = MCOperand::createFPImm(Imm->getValueAPF().convertToFloat());
      else if (Imm->getType()->isDoubleTy())
        MCOp = MCOperand::createFPImm(Imm->getValueAPF().convertToDouble());
      else
        llvm_unreachable("unknown floating point immediate type");
      break;
    }
    case MachineOperand::MO_MachineBasicBlock:
      MCOp = MCOperand::createExpr(
          MCSymbolRefExpr::create(MO.getMBB()->getSymbol(), Ctx));
      break;
    case MachineOperand::MO_GlobalAddress:
      MCOp = LowerSymbolOperand(MO, GetGlobalAddressSymbol(MO));
      break;
    case MachineOperand::MO_ExternalSymbol:
      MCOp = LowerSymbolOperand(MO, GetExternalSymbolSymbol(MO));
      break;
    }

    OutMI.addOperand(MCOp);
  }
}
