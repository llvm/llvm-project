//===- HexagonMCInstLower.cpp - Convert Hexagon MachineInstr to an MCInst -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower Hexagon MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "HexagonAsmPrinter.h"
#include "HexagonMachineFunctionInfo.h"
#include "llvm/Constants.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Target/Mangler.h"

using namespace llvm;

static MCOperand GetSymbolRef(const MachineOperand& MO, const MCSymbol* Symbol,
                              HexagonAsmPrinter& Printer) {
  MCContext &MC = Printer.OutContext;
  const MCExpr *ME;

  ME = MCSymbolRefExpr::Create(Symbol, MCSymbolRefExpr::VK_None, MC);

  if (!MO.isJTI() && MO.getOffset())
    ME = MCBinaryExpr::CreateAdd(ME, MCConstantExpr::Create(MO.getOffset(), MC),
                                 MC);

  return (MCOperand::CreateExpr(ME));
}

// Create an MCInst from a MachineInstr
void llvm::HexagonLowerToMC(const MachineInstr* MI, MCInst& MCI,
                            HexagonAsmPrinter& AP) {
  MCI.setOpcode(MI->getOpcode());

  for (unsigned i = 0, e = MI->getNumOperands(); i < e; i++) {
    const MachineOperand &MO = MI->getOperand(i);
    MCOperand MCO;

    switch (MO.getType()) {
    default:
      MI->dump();
      llvm_unreachable("unknown operand type");
    case MachineOperand::MO_Register:
      // Ignore all implicit register operands.
      if (MO.isImplicit()) continue;
      MCO = MCOperand::CreateReg(MO.getReg());
      break;
    case MachineOperand::MO_FPImmediate: {
      APFloat Val = MO.getFPImm()->getValueAPF();
      // FP immediates are used only when setting GPRs, so they may be dealt
      // with like regular immediates from this point on.
      MCO = MCOperand::CreateImm(*Val.bitcastToAPInt().getRawData());
      break;
    }
    case MachineOperand::MO_Immediate:
      MCO = MCOperand::CreateImm(MO.getImm());
      break;
    case MachineOperand::MO_MachineBasicBlock:
      MCO = MCOperand::CreateExpr
              (MCSymbolRefExpr::Create(MO.getMBB()->getSymbol(),
               AP.OutContext));
      break;
    case MachineOperand::MO_GlobalAddress:
      MCO = GetSymbolRef(MO, AP.Mang->getSymbol(MO.getGlobal()), AP);
      break;
    case MachineOperand::MO_ExternalSymbol:
      MCO = GetSymbolRef(MO, AP.GetExternalSymbolSymbol(MO.getSymbolName()),
                         AP);
      break;
    case MachineOperand::MO_JumpTableIndex:
      MCO = GetSymbolRef(MO, AP.GetJTISymbol(MO.getIndex()), AP);
      break;
    case MachineOperand::MO_ConstantPoolIndex:
      MCO = GetSymbolRef(MO, AP.GetCPISymbol(MO.getIndex()), AP);
      break;
    case MachineOperand::MO_BlockAddress:
      MCO = GetSymbolRef(MO, AP.GetBlockAddressSymbol(MO.getBlockAddress()),AP);
      break;
    }

    MCI.addOperand(MCO);
  }
}
