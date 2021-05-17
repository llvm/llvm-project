//===-- CSKYInstPrinter.cpp - Convert CSKY MCInst to asm syntax ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class prints an CSKY MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "CSKYInstPrinter.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

#define DEBUG_TYPE "csky-asm-printer"

// Include the auto-generated portion of the assembly writer.
#define PRINT_ALIAS_INSTR
#include "CSKYGenAsmWriter.inc"

static cl::opt<bool>
    NoAliases("csky-no-aliases",
              cl::desc("Disable the emission of assembler pseudo instructions"),
              cl::init(false), cl::Hidden);

static cl::opt<bool>
    ArchRegNames("csky-arch-reg-names",
                 cl::desc("Print architectural register names rather than the "
                          "ABI names (such as r14 instead of sp)"),
                 cl::init(false), cl::Hidden);

// The command-line flags above are used by llvm-mc and llc. They can be used by
// `llvm-objdump`, but we override their values here to handle options passed to
// `llvm-objdump` with `-M` (which matches GNU objdump). There did not seem to
// be an easier way to allow these options in all these tools, without doing it
// this way.
bool CSKYInstPrinter::applyTargetSpecificCLOption(StringRef Opt) {
  if (Opt == "no-aliases") {
    NoAliases = true;
    return true;
  }
  if (Opt == "numeric") {
    ArchRegNames = true;
    return true;
  }

  return false;
}

void CSKYInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                StringRef Annot, const MCSubtargetInfo &STI,
                                raw_ostream &O) {
  const MCInst *NewMI = MI;

  if (NoAliases || !printAliasInstr(NewMI, Address, STI, O))
    printInstruction(NewMI, Address, STI, O);
  printAnnotation(O, Annot);
}

void CSKYInstPrinter::printRegName(raw_ostream &O, unsigned RegNo) const {
  O << getRegisterName(RegNo);
}

void CSKYInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O,
                                   const char *Modifier) {
  assert((Modifier == 0 || Modifier[0] == 0) && "No modifiers supported");
  const MCOperand &MO = MI->getOperand(OpNo);

  if (MO.isReg()) {
    if (MO.getReg() == CSKY::C)
      O << "";
    else
      printRegName(O, MO.getReg());
    return;
  }

  if (MO.isImm()) {
    O << formatImm(MO.getImm());
    return;
  }

  assert(MO.isExpr() && "Unknown operand kind in printOperand");
  MO.getExpr()->print(O, &MAI);
}

const char *CSKYInstPrinter::getRegisterName(unsigned RegNo) {
  return getRegisterName(RegNo, ArchRegNames ? CSKY::NoRegAltName
                                             : CSKY::ABIRegAltName);
}
