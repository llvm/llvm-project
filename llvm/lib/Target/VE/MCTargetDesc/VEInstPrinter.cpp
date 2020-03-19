//===-- VEInstPrinter.cpp - Convert VE MCInst to assembly syntax -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class prints an VE MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "VEInstPrinter.h"
#include "VE.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "ve-asmprinter"

// The generated AsmMatcher VEGenAsmWriter uses "VE" as the target
// namespace.
namespace llvm {
namespace VE {
using namespace VE;
}
} // namespace llvm

#define GET_INSTRUCTION_NAME
#define PRINT_ALIAS_INSTR
#include "VEGenAsmWriter.inc"

void VEInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  OS << '%' << StringRef(getRegisterName(RegNo)).lower();
}

void VEInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                              StringRef Annot, const MCSubtargetInfo &STI,
                              raw_ostream &OS) {
  if (!printAliasInstr(MI, STI, OS))
    printInstruction(MI, Address, STI, OS);
  printAnnotation(OS, Annot);
}

void VEInstPrinter::printOperand(const MCInst *MI, int opNum,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(opNum);

  if (MO.isReg()) {
    printRegName(O, MO.getReg());
    return;
  }

  if (MO.isImm()) {
    switch (MI->getOpcode()) {
    default:
      // Expects signed 32bit literals
      int32_t TruncatedImm = static_cast<int32_t>(MO.getImm());
      O << TruncatedImm;
      return;
    }
  }

  assert(MO.isExpr() && "Unknown operand kind in printOperand");
  MO.getExpr()->print(O, &MAI);
}

void VEInstPrinter::printMemASXOperand(const MCInst *MI, int opNum,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &O, const char *Modifier) {
  // If this is an ADD operand, emit it like normal operands.
  if (Modifier && !strcmp(Modifier, "arith")) {
    printOperand(MI, opNum, STI, O);
    O << ", ";
    printOperand(MI, opNum + 1, STI, O);
    return;
  }

  const MCOperand &MO = MI->getOperand(opNum + 1);
  if (!MO.isImm() || MO.getImm() != 0) {
    printOperand(MI, opNum + 1, STI, O);
  }
  O << "(,";
  printOperand(MI, opNum, STI, O);
  O << ")";
}

void VEInstPrinter::printMemASOperand(const MCInst *MI, int opNum,
                                      const MCSubtargetInfo &STI,
                                      raw_ostream &O, const char *Modifier) {
  // If this is an ADD operand, emit it like normal operands.
  if (Modifier && !strcmp(Modifier, "arith")) {
    printOperand(MI, opNum, STI, O);
    O << ", ";
    printOperand(MI, opNum + 1, STI, O);
    return;
  }

  const MCOperand &MO = MI->getOperand(opNum + 1);
  if (!MO.isImm() || MO.getImm() != 0) {
    printOperand(MI, opNum + 1, STI, O);
  }
  O << "(";
  printOperand(MI, opNum, STI, O);
  O << ")";
}

void VEInstPrinter::printCCOperand(const MCInst *MI, int opNum,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  int CC = (int)MI->getOperand(opNum).getImm();
  O << VECondCodeToString((VECC::CondCodes)CC);
}
