//===- M88kInstPrinter.cpp - Convert M88k MCInst to assembly syntax -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "M88kInstPrinter.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#include "M88kGenAsmWriter.inc"

void M88kInstPrinter::printOperand(const MCInst *MI, int OpNum,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  if (MO.isReg()) {
    if (!MO.getReg())
      O << '0';
    else
      O << '%' << getRegisterName(MO.getReg());
  } else if (MO.isImm())
    O << MO.getImm();
  else if (MO.isExpr())
    MO.getExpr()->print(O, &MAI);
  else
    llvm_unreachable("Invalid operand");
}

void M88kInstPrinter::printOperand(const MCOperand &MO, const MCAsmInfo *MAI,
                                   raw_ostream &O) {
  if (MO.isReg()) {
    if (!MO.getReg())
      O << '0';
    else
      O << '%' << getRegisterName(MO.getReg());
  } else if (MO.isImm())
    O << MO.getImm();
  else if (MO.isExpr())
    MO.getExpr()->print(O, MAI);
  else
    llvm_unreachable("Invalid operand");
}

void M88kInstPrinter::printScaledRegister(const MCInst *MI, int OpNum,
                                          const MCSubtargetInfo &STI,
                                          raw_ostream &O) {
  assert(MI->getOperand(OpNum).isReg() && "Expected register");
  O << "[" << '%' << getRegisterName(MI->getOperand(OpNum).getReg()) << "]";
}

void M88kInstPrinter::printU5ImmOperand(const MCInst *MI, int OpNum,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  int64_t Value = MI->getOperand(OpNum).getImm();
  // assert(isUInt<N>(Value) && "Invalid uimm argument");
  O << Value;
}

void M88kInstPrinter::printU5ImmOOperand(const MCInst *MI, int OpNum,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  int64_t Value = MI->getOperand(OpNum).getImm();
  // assert(isUInt<N>(Value) && "Invalid uimm argument");
  O << "<" << Value << ">";
}

void M88kInstPrinter::printU16ImmOperand(const MCInst *MI, int OpNum,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  if (MO.isImm()) {
    O << MO.getImm();
  } else {
    assert(MO.isExpr() && "Expected expression");
    MO.getExpr()->print(O, &MAI);
  }
}

void M88kInstPrinter::printVec9Operand(const MCInst *MI, int OpNum,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  if (MO.isImm()) {
    O << MO.getImm();
  } else {
    assert(MO.isExpr() && "Expected expression");
    MO.getExpr()->print(O, &MAI);
  }
}

void M88kInstPrinter::printBitFieldOperand(const MCInst *MI, int OpNum,
                                           const MCSubtargetInfo &STI,
                                           raw_ostream &O) {
  int64_t Value = MI->getOperand(OpNum).getImm();
  assert(isUInt<10>(Value) && "Invalid bitfield argument");
  int64_t Width = (Value >> 5) & 0x1f;
  int64_t Offset = Value & 0x1f;
  O << Width << "<" << Offset << ">";
}

void M88kInstPrinter::printCCodeOperand(const MCInst *MI, int OpNum,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  assert(MO.isImm() && "Invalid condition code argument");
  int64_t CC = MO.getImm();
  switch (CC) {
#define CASE(cc, str)                                                          \
  case cc:                                                                     \
    O << str;                                                                  \
    break;
    CASE(0x2, "eq0")
    CASE(0xd, "ne0")
    CASE(0x1, "gt0")
    CASE(0xc, "lt0")
    CASE(0x3, "ge0")
    CASE(0xe, "le0")
#undef CASE
  default:
    O << CC;
  }
}

void M88kInstPrinter::printPCRelOperand(const MCInst *MI, uint64_t Address,
                                        int OpNum, const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  // TODO
  const MCOperand &MO = MI->getOperand(OpNum);
  if (MO.isImm()) {
    O << "0x";
    O.write_hex(MO.getImm());
  } else
    MO.getExpr()->print(O, &MAI);
}

void M88kInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                StringRef Annot, const MCSubtargetInfo &STI,
                                raw_ostream &O) {
  printInstruction(MI, Address, STI, O);
  printAnnotation(O, Annot);
}
