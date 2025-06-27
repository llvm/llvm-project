/* --- PEInstPrinter.cpp --- */

/* ------------------------------------------
author: 高宇翔
date: 4/9/2025
------------------------------------------ */

#include "PEInstPrinter.h"
#include "llvm/MC/MCExpr.h" // Include the header for MCExpr
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegister.h" // Include the header for MCRegister

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#define PRINT_ALIAS_INSTR
#include "PEGenAsmWriter.inc"

void PEInstPrinter::printRegName(raw_ostream &OS, MCRegister Reg) {
  OS << getRegisterName(Reg);
}

void PEInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                              StringRef Annot, const MCSubtargetInfo &STI,
                              raw_ostream &O) {
  // 判断是否为DIVR指令
  if (MI->getOpcode() == PE::DIVR) {
    // 假设操作数顺序为 rd, rs1, rs2
    O << "\tDIV ";
    printOperand(MI, 0, O); // rd
    O << ", ";
    printOperand(MI, 1, O); // rs1
    O << ", ";
    printOperand(MI, 2, O); // rs2
    O << "\n";
    O << "\tDIVR ";
    printOperand(MI, 0, O); // rd
    printAnnotation(O, Annot);
    return;
  }
  if (!printAliasInstr(MI, Address, O))
    printInstruction(MI, Address, O);
  printAnnotation(O, Annot);
}

void llvm::PEInstPrinter::PrintMemOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {

  // 打印格式：LD RD, RS,#imme
  // 先打印寄存器
  printOperand(MI, OpNo, O);
  O << ", ";
  // 再打印操作数
  if (OpNo + 1 < MI->getNumOperands()) {
    printOperand(MI, OpNo + 1, O);
  }
}

void llvm::PEInstPrinter::PrintVMemOperand(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  // 打印格式：LD RD, RS,#imme
  // 先打印寄存器
  printOperand(MI, OpNo, O);
  O << ", ";
  // 再打印操作数
  if (OpNo + 1 < MI->getNumOperands()) {
    printOperand(MI, OpNo + 1, O);
  }
}

void llvm::PEInstPrinter::PrintPtrOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  printOperand(MI, OpNo, O); // 打印第一个操作数
  // O << ", ";
  if (OpNo + 1 < MI->getNumOperands()) {
    O << "[";
    printOperand(MI, OpNo + 1, O); // 打印第二个操作数
    O << "]";
  }
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
void PEInstPrinter::printBranchOperand(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isExpr())
    Op.getExpr()->print(O, &MAI);
  else
    O << Op.getImm();
}
// const char *PEInstPrinter::getRegisterName(MCRegister Reg){
//     return getRegisterName(Reg);

// }