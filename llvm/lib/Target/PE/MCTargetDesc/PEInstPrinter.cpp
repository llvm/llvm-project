/* --- PEInstPrinter.cpp --- */

/* ------------------------------------------
author: 高宇翔
date: 4/9/2025
------------------------------------------ */

#include "PEInstPrinter.h"
#include "llvm/MC/MCRegister.h" // Include the header for MCRegister
#include "llvm/MC/MCInst.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#define PRINT_ALIAS_INSTR
#include "PEGenAsmWriter.inc"

void PEInstPrinter::printRegName(raw_ostream &OS, MCRegister Reg) {
  OS << StringRef(getRegisterName(Reg,PE::ABIRegAltName)).lower();
}

void PEInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                              StringRef Annot, const MCSubtargetInfo &STI,
                              raw_ostream &O) {
  if(!printAliasInstr(MI, Address, O))                              
  printInstruction(MI, Address, O);
  printAnnotation(O, Annot);
}

void llvm::PEInstPrinter::PrintMemOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  
//打印格式：12(sp)
//先打印立即数
printOperand(MI, OpNo+1, O);
O<<"(";
//再打印寄存器
printOperand(MI, OpNo, O);
O<<")";
}

void PEInstPrinter::printOperand(const MCInst *MI, unsigned OpNum,
                                 raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);
  if (Op.isReg()) {
    printRegName(O, Op.getReg());
    return;
  }

  if (Op.isImm()) {
    O << Op.getImm();
    return;
  }
}
const char *PEInstPrinter::getRegisterName(MCRegister Reg){
    return getRegisterName(Reg,PE::NoRegAltName);

}