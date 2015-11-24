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
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
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
  // Note that there's an implicit get_local/set_local here!
  OS << "$" << RegNo;
}

void WebAssemblyInstPrinter::printInst(const MCInst *MI, raw_ostream &OS,
                                       StringRef Annot,
                                       const MCSubtargetInfo &STI) {
  printInstruction(MI, OS);

  const MCInstrDesc &Desc = MII.get(MI->getOpcode());
  if (Desc.isVariadic())
    for (unsigned i = Desc.getNumOperands(), e = MI->getNumOperands(); i < e;
         ++i) {
      if (i != 0)
        OS << ", ";
      printOperand(MI, i, OS);
    }

  printAnnotation(OS, Annot);
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
    unsigned WAReg = Op.getReg();
    if (int(WAReg) >= 0)
      printRegName(O, WAReg);
    else if (OpNo >= MII.get(MI->getOpcode()).getNumDefs())
      O << "$pop" << (WAReg & INT32_MAX);
    else if (WAReg != WebAssemblyFunctionInfo::UnusedReg)
      O << "$push" << (WAReg & INT32_MAX);
    else
      O << "$discard";
    // Add a '=' suffix if this is a def.
    if (OpNo < MII.get(MI->getOpcode()).getNumDefs())
      O << '=';
  } else if (Op.isImm()) {
    switch (MI->getOpcode()) {
    case WebAssembly::PARAM:
    case WebAssembly::RESULT:
    case WebAssembly::LOCAL:
      switch (Op.getImm()) {
      case MVT::i32: O << "i32"; break;
      case MVT::i64: O << "i64"; break;
      case MVT::f32: O << "f32"; break;
      case MVT::f64: O << "f64"; break;
      default: llvm_unreachable("unexpected type");
      }
      break;
    default:
      O << Op.getImm();
      break;
    }
  } else if (Op.isFPImm())
    O << toString(APFloat(Op.getFPImm()));
  else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    Op.getExpr()->print(O, &MAI);
  }
}
