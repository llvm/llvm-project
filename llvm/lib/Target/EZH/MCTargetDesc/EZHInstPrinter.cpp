//===-- EZHInstPrinter.cpp - Convert EZH MCInst to asm syntax ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EZHInstPrinter.h"
#include "EZHMCTargetDesc.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#include "EZHGenAsmWriter.inc"

void EZHInstPrinter::printRegName(raw_ostream &OS, MCRegister Reg) {
  OS << StringRef(getRegisterName(Reg)).lower();
}

void EZHInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                               StringRef Annotation, const MCSubtargetInfo &STI,
                               raw_ostream &OS) {
  StringRef Name = MII.getName(MI->getOpcode());
  if (Name.starts_with("STR_PRE") && MI->getNumOperands() >= 4 &&
      MI->getOperand(0).isReg() && MI->getOperand(0).getReg() == EZH::SP &&
      MI->getOperand(2).isReg() && MI->getOperand(2).getReg() == EZH::SP &&
      MI->getOperand(3).isImm() && MI->getOperand(3).getImm() == -4) {
    OS << "\te_pushd" << Name.substr(7) << "\t";
    printOperand(MI, 1, OS);
    printAnnotation(OS, Annotation);
    return;
  }

  if (Name.starts_with("LDR_POST") && MI->getNumOperands() >= 4 &&
      MI->getOperand(1).isReg() && MI->getOperand(1).getReg() == EZH::SP &&
      MI->getOperand(2).isReg() && MI->getOperand(2).getReg() == EZH::SP &&
      MI->getOperand(3).isImm() && MI->getOperand(3).getImm() == 4) {
    OS << "\te_popd" << Name.substr(8) << "\t";
    printOperand(MI, 0, OS);
    printAnnotation(OS, Annotation);
    return;
  }

  printInstruction(MI, Address, OS);
  printAnnotation(OS, Annotation);
}

void EZHInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &OS) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg())
    OS << getRegisterName(Op.getReg());
  else if (Op.isImm())
    OS << Op.getImm();
  else {
    assert(Op.isExpr() && "Expected an expression");
    MAI.printExpr(OS, *Op.getExpr());
  }
}

void EZHInstPrinter::printBranchTarget(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    // Print the absolute address as a hex value without the 0x prefix for
    // objdump compatibility, or with 0x depending on conventions. Wait,
    // format_hex prints 0x. We can just use it.
    O << format_hex(Op.getImm(), 2);
  } else {
    printOperand(MI, OpNo, O);
  }
}

void EZHInstPrinter::printWordOffset(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    O << Op.getImm();
  } else {
    printOperand(MI, OpNo, O);
  }
}

void EZHInstPrinter::printImmOperand(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O) {
  printOperand(MI, OpNo, O);
}

void EZHInstPrinter::printShiftedImmOperand(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) {
  const MCOperand &ImmOp = MI->getOperand(OpNo);
  const MCOperand &ShAmtOp = MI->getOperand(OpNo + 1);

  if (ImmOp.isImm() && ShAmtOp.isImm()) {
    int64_t Imm = ImmOp.getImm();
    int64_t ShAmt = ShAmtOp.getImm();
    O << (Imm << ShAmt);
  } else {
    // Fallback if not an immediate
    printOperand(MI, OpNo, O);
  }
}
