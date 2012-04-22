//===- HexagonInstPrinter.cpp - Convert Hexagon MCInst to assembly syntax -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an Hexagon MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "Hexagon.h"
#include "HexagonAsmPrinter.h"
#include "HexagonInstPrinter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>

using namespace llvm;

#define GET_INSTRUCTION_NAME
#include "HexagonGenAsmWriter.inc"

StringRef HexagonInstPrinter::getOpcodeName(unsigned Opcode) const {
  return MII.getName(Opcode);
}

StringRef HexagonInstPrinter::getRegName(unsigned RegNo) const {
  return getRegisterName(RegNo);
}

void HexagonInstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                                   StringRef Annot) {
  const char packetPadding[] = "      ";
  const char startPacket = '{',
             endPacket = '}';
  // TODO: add outer HW loop when it's supported too.
  if (MI->getOpcode() == Hexagon::ENDLOOP0) {
    MCInst Nop;

    O << packetPadding << startPacket << '\n';
    Nop.setOpcode(Hexagon::NOP);
    printInstruction(&Nop, O);
    O << packetPadding << endPacket;
  }

  printInstruction(MI, O);
  printAnnotation(O, Annot);
}

void HexagonInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O) const {
  const MCOperand& MO = MI->getOperand(OpNo);

  if (MO.isReg()) {
    O << getRegisterName(MO.getReg());
  } else if(MO.isExpr()) {
    O << *MO.getExpr();
  } else if(MO.isImm()) {
    printImmOperand(MI, OpNo, O);
  } else {
    assert(false && "Unknown operand");
  }
}

void HexagonInstPrinter::printImmOperand
  (const MCInst *MI, unsigned OpNo, raw_ostream &O) const {
  O << MI->getOperand(OpNo).getImm();
}

void HexagonInstPrinter::printExtOperand(const MCInst *MI, unsigned OpNo,
                                                raw_ostream &O) const {
  O << MI->getOperand(OpNo).getImm();
}

void HexagonInstPrinter::printUnsignedImmOperand
  (const MCInst *MI, unsigned OpNo, raw_ostream &O) const {
  O << MI->getOperand(OpNo).getImm();
}

void HexagonInstPrinter::printNegImmOperand(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) const {
  O << -MI->getOperand(OpNo).getImm();
}

void HexagonInstPrinter::printNOneImmOperand
  (const MCInst *MI, unsigned OpNo, raw_ostream &O) const {
  O << -1;
}

void HexagonInstPrinter::printMEMriOperand
  (const MCInst *MI, unsigned OpNo, raw_ostream &O) const {
  const MCOperand& MO0 = MI->getOperand(OpNo);
  const MCOperand& MO1 = MI->getOperand(OpNo + 1);

  O << getRegisterName(MO0.getReg());
  O << " + #" << MO1.getImm();
}

void HexagonInstPrinter::printFrameIndexOperand
  (const MCInst *MI, unsigned OpNo, raw_ostream &O) const {
  const MCOperand& MO0 = MI->getOperand(OpNo);
  const MCOperand& MO1 = MI->getOperand(OpNo + 1);

  O << getRegisterName(MO0.getReg()) << ", #" << MO1.getImm();
}

void HexagonInstPrinter::printGlobalOperand(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) const {
  assert(MI->getOperand(OpNo).isExpr() && "Expecting expression");

  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printJumpTable(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) const {
  assert(MI->getOperand(OpNo).isExpr() && "Expecting expression");

  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printConstantPool(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) const {
  assert(MI->getOperand(OpNo).isExpr() && "Expecting expression");

  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printBranchOperand(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) const {
  // Branches can take an immediate operand.  This is used by the branch
  // selection pass to print $+8, an eight byte displacement from the PC.
  assert("Unknown branch operand.");
}

void HexagonInstPrinter::printCallOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) const {
}

void HexagonInstPrinter::printAbsAddrOperand(const MCInst *MI, unsigned OpNo,
                                             raw_ostream &O) const {
}

void HexagonInstPrinter::printPredicateOperand(const MCInst *MI, unsigned OpNo,
                                               raw_ostream &O) const {
}

void HexagonInstPrinter::printSymbol(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O, bool hi) const {
  const MCOperand& MO = MI->getOperand(OpNo);

  O << '#' << (hi? "HI": "LO") << '(';
  if (MO.isImm()) {
    O << '#';
    printOperand(MI, OpNo, O);
  } else {
    assert("Unknown symbol operand");
    printOperand(MI, OpNo, O);
  }
  O << ')';
}
