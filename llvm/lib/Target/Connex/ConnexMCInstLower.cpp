//=-- ConnexMCInstLower.cpp - Convert Connex MachineInstr to an MCInst ------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower Connex MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "ConnexMCInstLower.h"
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

#include "llvm/Support/Debug.h" // for dbgs and LLVM_DEBUG() macro
#define DEBUG_TYPE "mc-inst-lower"

using namespace llvm;

MCSymbol *
ConnexMCInstLower::GetGlobalAddressSymbol(const MachineOperand &MO) const {
  return Printer.getSymbol(MO.getGlobal());
}

MCSymbol *
ConnexMCInstLower::GetExternalSymbolSymbol(const MachineOperand &MO) const {
  return Printer.GetExternalSymbolSymbol(MO.getSymbolName());
}

MCOperand ConnexMCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                                MCSymbol *Sym) const {

  const MCExpr *Expr = MCSymbolRefExpr::create(Sym, Ctx);

  if (!MO.isJTI() && MO.getOffset())
    llvm_unreachable("unknown symbol op");

  return MCOperand::createExpr(Expr);
}

void ConnexMCInstLower::Lower(const MachineInstr *MI, MCInst &OutMI) const {
  LLVM_DEBUG(dbgs() << "Entered ConnexMCInstLower::Lower(*MI = " << *MI
                    << ")...\n");
  OutMI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    LLVM_DEBUG(dbgs() << "ConnexMCInstLower::Lower(): MO = " << MO << "\n");
    LLVM_DEBUG(dbgs() << "  ConnexMCInstLower::Lower(): MO.getType() = "
                      << MO.getType() << "\n");

    MCOperand MCOp;

    switch (MO.getType()) {

    default:
      MI->dump();
      /*
      LLVM_DEBUG(dbgs() << "ConnexMCInstLower::Lower(): MO.getType() = "
                        << MO.getType() << "\n");
      */

      llvm_unreachable("unknown operand type");

    case MachineOperand::MO_ExternalSymbol: {
      // MEGA-MEGA-TODO: check if OK
      /*
      const MCSymbol *Symbol =
          Printer.GetExternalSymbolSymbol(MO.getSymbolName());
      MCSymbolRefExpr::VariantKind Kind = MCSymbolRefExpr::VK_None;
      const MCExpr *Expr = MCSymbolRefExpr::create(Symbol, Kind, Ctx);
      MCOp = MCOperand::createExpr(Expr);
      // Offset += MO.getOffset();
      */
      // Inspired from BPFMCInstLower.cpp (from Oct 2025)
      MCOp = LowerSymbolOperand(MO, GetExternalSymbolSymbol(MO));

      break;
    }

    // case MachineOperand::MO_MetaData:
    case MachineOperand::MO_Metadata: {
      continue;
      // break;
    }

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
    }

    OutMI.addOperand(MCOp);
  }
}
