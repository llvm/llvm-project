//=- SystemZHLASMInstPrinter.cpp - Convert SystemZ MCInst to HLASM assembly -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZHLASMInstPrinter.h"
#include "SystemZInstrInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#include "SystemZGenHLASMAsmWriter.inc"

void SystemZHLASMInstPrinter::printFormattedRegName(const MCAsmInfo *MAI,
                                                    MCRegister Reg,
                                                    raw_ostream &O) {
  const char *RegName = getRegisterName(Reg);
  // Skip register prefix so that only register number is left
  assert(isalpha(RegName[0]) && isdigit(RegName[1]));
  markup(O, Markup::Register) << (RegName + 1);
}

void SystemZHLASMInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                        StringRef Annot,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  std::string Str;
  raw_string_ostream RSO(Str);
  printInstruction(MI, Address, RSO);
  // Eat the first tab character and replace it with a space since it is
  // hardcoded in AsmWriterEmitter::EmitPrintInstruction.
  // TODO Introduce a line prefix member to AsmWriter to avoid this problem.
  if (!Str.empty() && Str.front() == '\t')
    O << " " << Str.substr(1, Str.length());
  else
    O << Str;

  printAnnotation(O, Annot);
}

void SystemZHLASMInstPrinter::printPCRelOperand(const MCInst *MI, int OpNum,
                                                raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  if (MO.isImm()) {
    WithMarkup M = markup(O, Markup::Immediate);
    O << "0x";
    O.write_hex(MO.getImm());
  } else {
    // Don't print @PLT.
    // TODO May need to do something more here.
    const MCSymbolRefExpr &SRE = cast<MCSymbolRefExpr>(*(MO.getExpr()));
    O << SRE.getSymbol().getName();
  }
}
