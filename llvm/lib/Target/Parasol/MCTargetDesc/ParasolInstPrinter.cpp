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

    // If the immediate is 32 bit (I think we only have this at least for now)
    // and could be negative in 2's complement, also print that out
    uint32_t trunc_imm = (uint32_t)Op.getImm();
    if (trunc_imm == Op.getImm() && (trunc_imm & 0x80000000)) {
      O << " (" << (int32_t)trunc_imm << ")";
    }

    // If it's branching, also display the number of instructions involved
    switch (MI->getOpcode()) {
    case Parasol::BR:
    case Parasol::BRZ:
    case Parasol::BRNZ:
      int32_t inst_count = ((int32_t)trunc_imm) / 8;
      if (inst_count > 0) {
        O << " [Down " << inst_count << "]";
      } else {
        O << " [Up " << -inst_count << "]";
      }
      break;
    }

    return;
  }

  assert(Op.isExpr() && "unknown operand kind in printOperand");
  Op.getExpr()->print(O, &MAI, true);
}
