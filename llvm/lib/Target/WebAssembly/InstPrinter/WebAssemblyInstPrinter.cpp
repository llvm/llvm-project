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
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
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
                                       const MCSubtargetInfo & /*STI*/) {
  // Print the instruction (this uses the AsmStrings from the .td files).
  printInstruction(MI, OS);

  // Print any additional variadic operands.
  const MCInstrDesc &Desc = MII.get(MI->getOpcode());
  if (Desc.isVariadic())
    for (auto i = Desc.getNumOperands(), e = MI->getNumOperands(); i < e; ++i) {
      if (i != 0)
        OS << ", ";
      printOperand(MI, i, OS);
    }

  // Print any added annotation.
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
    assert((OpNo < MII.get(MI->getOpcode()).getNumOperands() ||
            MII.get(MI->getOpcode()).TSFlags == 0) &&
           "WebAssembly variable_ops register ops don't use TSFlags");
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
    assert((OpNo < MII.get(MI->getOpcode()).getNumOperands() ||
            (MII.get(MI->getOpcode()).TSFlags &
                     WebAssemblyII::VariableOpIsImmediate)) &&
           "WebAssemblyII::VariableOpIsImmediate should be set for "
           "variable_ops immediate ops");
    if (MII.get(MI->getOpcode()).TSFlags &
        WebAssemblyII::VariableOpImmediateIsType)
      // The immediates represent types.
      O << WebAssembly::TypeToString(MVT::SimpleValueType(Op.getImm()));
    else
      O << Op.getImm();
  } else if (Op.isFPImm()) {
    assert((OpNo < MII.get(MI->getOpcode()).getNumOperands() ||
            MII.get(MI->getOpcode()).TSFlags == 0) &&
           "WebAssembly variable_ops floating point ops don't use TSFlags");
    O << toString(APFloat(Op.getFPImm()));
  } else {
    assert((OpNo < MII.get(MI->getOpcode()).getNumOperands() ||
            (MII.get(MI->getOpcode()).TSFlags &
                     WebAssemblyII::VariableOpIsImmediate)) &&
           "WebAssemblyII::VariableOpIsImmediate should be set for "
           "variable_ops expr ops");
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    Op.getExpr()->print(O, &MAI);
  }
}

const char *llvm::WebAssembly::TypeToString(MVT Ty) {
  switch (Ty.SimpleTy) {
  case MVT::i32:
    return "i32";
  case MVT::i64:
    return "i64";
  case MVT::f32:
    return "f32";
  case MVT::f64:
    return "f64";
  default:
    llvm_unreachable("unsupported type");
  }
}
