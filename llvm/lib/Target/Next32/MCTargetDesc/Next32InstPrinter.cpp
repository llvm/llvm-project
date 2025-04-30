//===-- Next32InstPrinter.cpp - Convert Next32 MCInst to asm syntax--------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an Next32 MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "Next32InstPrinter.h"
#include "MCTargetDesc/Next32MCExpr.h"
#include "Next32.h"
#include "Next32InstrInfo.h"
#include "TargetInfo/Next32BaseInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

// Include the auto-generated portion of the assembly writer.
#include "Next32GenAsmWriter.inc"

void Next32InstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                  StringRef Annot, const MCSubtargetInfo &STI,
                                  raw_ostream &OS) {
  printInstruction(MI, Address, OS);
  printAnnotation(OS, Annot);
}

void Next32InstPrinter::printExpr(const MCExpr *Expr, raw_ostream &O) {

  if (auto *SymExpr = dyn_cast<MCSymbolRefExpr>(Expr)) {
    O << SymExpr->getSymbol().getName();
    return;
  }
  if (auto *Next32Expr = dyn_cast<Next32MCExpr>(Expr)) {
    Next32Expr->printImpl(O, &MAI);
    return;
  }
  llvm_unreachable("Unsupported MCExpr");
}

void Next32InstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O, const char *Modifier) {
  assert((Modifier == 0 || Modifier[0] == 0) && "No modifiers supported");
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    O << getRegisterName(Op.getReg());
  } else if (Op.isImm()) {
    O << "0x" << utohexstr((uint32_t)Op.getImm());
  } else {
    assert(Op.isExpr() && "Expected an expression");
    printExpr(Op.getExpr(), O);
  }
}

void Next32InstPrinter::printCondition(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  O << Next32Helpers::GetCondCodeString((Next32Constants::CondCode)Op.getImm());
}

void Next32InstPrinter::printInstructionSize(const MCInst *MI, unsigned OpNo,
                                             raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (!Op.isImm())
    return;

  O << "." << Next32Helpers::SizeFieldValueToBits(Op.getImm());
}

void Next32InstPrinter::printVectorElementCount(const MCInst *MI, unsigned OpNo,
                                                raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (!Op.isImm())
    return;

  if (!Op.getImm())
    return;

  O << "." << Next32Helpers::Log2VecElemFieldValueToCount(Op.getImm());
}

void Next32InstPrinter::printMemoryAddressSpace(const MCInst *MI, unsigned OpNo,
                                                raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (!Op.isImm())
    return;

  switch (Op.getImm()) {
  case Next32Constants::InstCodeAddressSpace::GENERIC:
    break;
  case Next32Constants::InstCodeAddressSpace::TLS:
    O << ".tls";
    break;
  case Next32Constants::InstCodeAddressSpace::GLOBAL:
    O << ".global";
    break;
  case Next32Constants::InstCodeAddressSpace::CONST:
    O << ".const";
    break;
  case Next32Constants::InstCodeAddressSpace::LOCAL:
    O << ".local";
    break;
  default:
    report_fatal_error("Unexpected Address Space Instruction Code!");
  }
}

void Next32InstPrinter::printAlignSize(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (!Op.isImm())
    return;

  O << ".align[" << Next32Helpers::Log2AlignValueToBytes(Op.getImm()) << "]";
}

void Next32InstPrinter::printRRIAttribute(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isImm() && "Expected an immediate");
  O << Next32Helpers::GetRRIAttributeMnemonic(Op.getImm());
}
