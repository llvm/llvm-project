/* --- PEAsmPrinter.cpp --- */

/* ------------------------------------------
author: undefined
date: 4/9/2025
------------------------------------------ */

#include "PEAsmPrinter.h"
#include "MCTargetDesc/PEMCExpr.h"
#include "MCTargetDesc/PEMCTargetDesc.h"
using namespace llvm;
#define DEBUG_TYPE "asm-printer"

#include "PEGenMCPseudoLowering.inc"
bool PEAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  AsmPrinter::runOnMachineFunction(MF);
  return true;
}

void PEAsmPrinter::emitInstruction(const MachineInstr *MI) {

  if (MCInst OutInst; lowerPseudoInstExpansion(MI, OutInst)) {
    EmitToStreamer(*OutStreamer, OutInst);
    return;
  }
  MCInst TmpInst;
  lowerToMCInst(MI, TmpInst);
  EmitToStreamer(*OutStreamer, TmpInst);
}

bool PEAsmPrinter::lowerToMCInst(const MachineInstr *MI, MCInst &OutMI) {
  OutMI.setOpcode(MI->getOpcode());
  for (const MachineOperand &MO : MI->operands()) {
    MCOperand MCOp;
    switch (MO.getType()) {
    case MachineOperand::MO_Register: {
      MCOp = MCOperand::createReg(MO.getReg());
      break;
    }
    case MachineOperand::MO_Immediate: {
      MCOp = MCOperand::createImm(MO.getImm());
      break;
    }
    case MachineOperand::MO_FrameIndex: {
      MCOp = MCOperand::createImm(MO.getIndex());
      break;
    }
    case MachineOperand::MO_GlobalAddress:
    case MachineOperand::MO_MachineBasicBlock: 
    {
      MCOp = lowerSymbolOperand(MO);
      break;
    }
    default: {
      llvm_unreachable("Unhandled operand type");
      break;
    }
    }
    OutMI.addOperand(MCOp);
  }
  return true;
}

MCOperand PEAsmPrinter::lowerSymbolOperand(const MachineOperand &MO) const {
  //   auto *symbol = getSymbol(MO.getGlobal());
  //   const auto &expr =
  //       MCSymbolRefExpr::create(symbol, MCSymbolRefExpr::VK_None,
  //       OutContext);
  //   return MCOperand::createExpr(expr);
  PEMCExpr::Kind Kind = PEMCExpr::NONE;
  const MCSymbol *Symbol = nullptr;
  switch (MO.getTargetFlags()) {
  case PEMCExpr::HI:
    Kind = PEMCExpr::HI;
    break;
  case PEMCExpr::LO:
    Kind = PEMCExpr::LO;
    break;
  default:
    break;
  }
  if(MO.getType() == MachineOperand::MO_MachineBasicBlock)
  {
    Symbol = MO.getMBB()->getSymbol();
  }
  else{
    Symbol = getSymbol(MO.getGlobal());
  }
  const MCExpr *Expr = MCSymbolRefExpr::create(Symbol, OutContext);
  Expr = new PEMCExpr(Kind, Expr);
  return MCOperand::createExpr(Expr);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializePEAsmPrinter() {
  RegisterAsmPrinter<PEAsmPrinter> X(getPETarget());
}