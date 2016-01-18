//=- WebAssemblyInstPrinter.cpp - WebAssembly assembly instruction printing -=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Print MCInst instructions to wasm format.
///
//===----------------------------------------------------------------------===//

#include "InstPrinter/WebAssemblyInstPrinter.h"
#include "WebAssembly.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <cctype>
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#include "WebAssemblyGenAsmWriter.inc"

WebAssemblyInstPrinter::WebAssemblyInstPrinter(const MCAsmInfo &MAI,
                                               const MCInstrInfo &MII,
                                               const MCRegisterInfo &MRI)
    : MCInstPrinter(MAI, MII, MRI) {}

void WebAssemblyInstPrinter::printRegName(raw_ostream &OS,
                                          unsigned RegNo) const {
  if (TargetRegisterInfo::isPhysicalRegister(RegNo))
    OS << getRegisterName(RegNo);
  else
    OS << TargetRegisterInfo::virtReg2Index(RegNo);
}

void WebAssemblyInstPrinter::printInst(const MCInst *MI, raw_ostream &OS,
                                       StringRef Annot,
                                       const MCSubtargetInfo &STI) {
  printInstruction(MI, OS);
  printAnnotation(OS, Annot);

  unsigned NumDefs = MII.get(MI->getOpcode()).getNumDefs();
  assert(NumDefs <= 1 &&
         "Instructions with multiple result values not implemented");

  if (NumDefs != 0) {
    OS << "\n"
          "\t" "set_local ";
    printRegName(OS, MI->getOperand(0).getReg());
    OS << ", pop";
  }
}

void WebAssemblyInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    if (OpNo < MII.get(MI->getOpcode()).getNumDefs())
      O << "push";
    else
      printRegName(O, Op.getReg());
  } else if (Op.isImm())
    O << '#' << Op.getImm();
  else if (Op.isFPImm())
    O << '#' << Op.getFPImm();
  else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    Op.getExpr()->print(O, &MAI);
  }
}
