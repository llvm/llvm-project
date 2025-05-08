//===- ParasolInstPrinter.cpp - Convert Parasol MCInst to assembly syntax -===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This class prints an Parasol MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "ParasolInstPrinter.h"

#include "ParasolInstrInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "parasol-isel"

#define PRINT_ALIAS_INSTR
#include "ParasolGenAsmWriter.inc"

ParasolInstPrinter::ParasolInstPrinter(const MCAsmInfo &MAI,
                                       const MCInstrInfo &MII,
                                       const MCRegisterInfo &MRI)
    : MCInstPrinter(MAI, MII, MRI) {}

void ParasolInstPrinter::printRegName(raw_ostream &OS, MCRegister RegNo) const {
  OS << StringRef(getRegisterName(RegNo)).lower();
}

void ParasolInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                   StringRef Annot, const MCSubtargetInfo &STI,
                                   raw_ostream &O) {
  // Try to print any aliases first.
  if (!printAliasInstr(MI, Address, O)) {
    printInstruction(MI, Address, O);
  }
  printAnnotation(O, Annot);
}

void ParasolInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
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
