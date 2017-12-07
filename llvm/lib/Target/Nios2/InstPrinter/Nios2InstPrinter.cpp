//===-- Nios2InstPrinter.cpp - Convert Nios2 MCInst to assembly syntax-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an Nios2 MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "Nios2InstPrinter.h"

#include "Nios2InstrInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#define PRINT_ALIAS_INSTR
#include "Nios2GenAsmWriter.inc"

void Nios2InstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  OS << getRegisterName(RegNo);
}

void Nios2InstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                                 StringRef Annot, const MCSubtargetInfo &STI) {
  // Try to print any aliases first.
  if (!printAliasInstr(MI, STI, O))
    printInstruction(MI, STI, O);
  printAnnotation(O, Annot);
}

void Nios2InstPrinter::printOperand(const MCInst *MI, int OpNo,
                                    const MCSubtargetInfo &STI,
                                    raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    printRegName(O, Op.getReg());
    return;
  }

  if (Op.isImm()) {
    O << Op.getImm();
    return;
  }

  assert(Op.isExpr() && "unknown operand kind in printOperand");
  Op.getExpr()->print(O, &MAI, true);
}

void Nios2InstPrinter::printMemOperand(const MCInst *MI, int opNum,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &O, const char *Modifier) {
  // Load/Store memory operands -- imm($reg)
  printOperand(MI, opNum + 1, STI, O);
  O << "(";
  printOperand(MI, opNum, STI, O);
  O << ")";
}
