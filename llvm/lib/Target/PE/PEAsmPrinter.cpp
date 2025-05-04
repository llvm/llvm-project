/* --- PEAsmPrinter.cpp --- */

/* ------------------------------------------
author: undefined
date: 4/9/2025
------------------------------------------ */

#include "PEAsmPrinter.h"
#include "MCTargetDesc/PEMCTargetDesc.h"
using namespace llvm;
#define DEBUG_TYPE "asm-printer"

#include "PEGenMCPseudoLowering.inc"
bool PEAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
    AsmPrinter::runOnMachineFunction(MF);
    return true;
}

void PEAsmPrinter::emitInstruction(const MachineInstr *MI) {

    if(MCInst OutInst;lowerPseudoInstExpansion(MI, OutInst)){
        EmitToStreamer(*OutStreamer, OutInst);
        return;
    }
    MCInst TmpInst;
    lowerToMCInst(MI, TmpInst);
    EmitToStreamer(*OutStreamer, TmpInst);
}

bool PEAsmPrinter::lowerToMCInst(const MachineInstr *MI, MCInst &OutMI){
    OutMI.setOpcode(MI->getOpcode());
    for (const MachineOperand &MO : MI->operands()) {
        MCOperand MCOp;
            switch (MO.getType())
            {
            case MachineOperand::MO_Register:{
                MCOp = MCOperand::createReg(MO.getReg());
                break;
            }
            case MachineOperand::MO_Immediate:{
                MCOp = MCOperand::createImm(MO.getImm());
                break;
            }
            default:
             llvm_unreachable("Unhandled operand type");
                break;
            }
            OutMI.addOperand(MCOp);
        }
        return true;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializePEAsmPrinter() {
    RegisterAsmPrinter<PEAsmPrinter> X(getPETarget());
  }
  