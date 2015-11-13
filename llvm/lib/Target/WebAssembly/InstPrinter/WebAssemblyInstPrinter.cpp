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
#include "WebAssemblyMachineFunctionInfo.h"
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
  assert(RegNo != WebAssemblyFunctionInfo::UnusedReg);
  // FIXME: Revisit whether we actually print the get_local explicitly.
  OS << "(get_local " << RegNo << ")";
}

void WebAssemblyInstPrinter::printInst(const MCInst *MI, raw_ostream &OS,
                                       StringRef Annot,
                                       const MCSubtargetInfo &STI) {
  printInstruction(MI, OS);

  const MCInstrDesc &Desc = MII.get(MI->getOpcode());
  if (Desc.isVariadic())
    for (unsigned i = Desc.getNumOperands(), e = MI->getNumOperands(); i < e;
         ++i) {
      OS << ", ";
      printOperand(MI, i, OS);
    }

  printAnnotation(OS, Annot);

  unsigned NumDefs = MII.get(MI->getOpcode()).getNumDefs();
  assert(NumDefs <= 1 &&
         "Instructions with multiple result values not implemented");

  // FIXME: Revisit whether we actually print the set_local explicitly.
  if (NumDefs != 0) {
    unsigned WAReg = MI->getOperand(0).getReg();
    // Only print the set_local if the register is used.
    // TODO: Revisit this once the spec explains what should happen here.
    if (WAReg != WebAssemblyFunctionInfo::UnusedReg)
      OS << "\n"
            "\t" "set_local " << WAReg << ", $pop";
  }
}

static std::string toString(const APFloat &FP) {
  static const size_t BufBytes = 128;
  char buf[BufBytes];
  if (FP.isNaN())
    assert((FP.bitwiseIsEqual(APFloat::getQNaN(FP.getSemantics())) ||
            FP.bitwiseIsEqual(
                APFloat::getQNaN(FP.getSemantics(), /*Negative=*/true))) &&
           "convertToHexString handles neither SNaN nor NaN payloads");
  // Use C99's hexadecimal floating-point representation.
  auto Written = FP.convertToHexString(
      buf, /*hexDigits=*/0, /*upperCase=*/false, APFloat::rmNearestTiesToEven);
  (void)Written;
  assert(Written != 0);
  assert(Written < BufBytes);
  return buf;
}

void WebAssemblyInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    if (OpNo >= MII.get(MI->getOpcode()).getNumDefs())
      printRegName(O, Op.getReg());
    else {
      if (Op.getReg() != WebAssemblyFunctionInfo::UnusedReg)
        O << "$push";
      else
        O << "$discard";
    }
  } else if (Op.isImm())
    O << Op.getImm();
  else if (Op.isFPImm())
    O << toString(APFloat(Op.getFPImm()));
  else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    Op.getExpr()->print(O, &MAI);
  }
}
