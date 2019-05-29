//===-- DPUInstPrinter.cpp - Convert DPU MCInst to asm syntax -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an DPU MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "DPUInstPrinter.h"
#include "MCTargetDesc/DPUAsmCondition.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include <llvm/Support/Debug.h>

using namespace llvm;

#include "llvm/CodeGen/ISDOpcodes.h"

#define DEBUG_TYPE "asm-printer"

#include "DPUGenAsmWriter.inc"

#include "DPUCondCodes.h"

void DPUInstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                               StringRef Annot, const MCSubtargetInfo &STI) {
  printInstruction(MI, O);
  printAnnotation(O, Annot);
}

static void printExpr(const MCExpr *Expr, raw_ostream &O) {
  const MCSymbolRefExpr *SRE;

  if (const auto *BE = dyn_cast<MCBinaryExpr>(Expr))
    SRE = dyn_cast<MCSymbolRefExpr>(BE->getLHS());
  else
    SRE = dyn_cast<MCSymbolRefExpr>(Expr);
  assert(SRE && "Unexpected MCExpr type.");

  if (SRE->getKind() != MCSymbolRefExpr::VK_None) {
    llvm_unreachable("Unexpected MCExpr");
  }

  O << *Expr;
}

void DPUInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O, const char *Modifier) {
  assert((Modifier == nullptr || Modifier[0] == 0) && "No modifiers supported");
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    O << getRegisterName(Op.getReg());
  } else if (Op.isImm()) {
    O << "0x";
    O.write_hex((uint64_t)Op.getImm());
  } else {
    assert(Op.isExpr() && "Expected an expression");
    printExpr(Op.getExpr(), O);
  }
}

template <unsigned Bits>
void DPUInstPrinter::printSImm(const MCInst *MI, unsigned OpNo, raw_ostream &O,
                               const char *Modifier) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    O << formatImm(Op.getImm());
  } else if (Op.isExpr()) {
    Op.getExpr()->print(O, &MAI);
  } else {
    llvm_unreachable(
        "print immediate invoked with non-immediate/expression operand");
  }
}

template <unsigned Bits>
void DPUInstPrinter::printUImm(const MCInst *MI, unsigned OpNo, raw_ostream &O,
                               const char *Modifier) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    O << formatImm(Op.getImm());
  } else if (Op.isExpr()) {
    Op.getExpr()->print(O, &MAI);
  } else {
    llvm_unreachable(
        "print immediate invoked with non-immediate/expression operand");
  }
}

template <unsigned Bits>
void DPUInstPrinter::printSUImm(const MCInst *MI, unsigned OpNo, raw_ostream &O,
                                const char *Modifier) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    O << formatImm(Op.getImm());
  } else if (Op.isExpr()) {
    Op.getExpr()->print(O, &MAI);
  } else {
    llvm_unreachable(
        "print immediate invoked with non-immediate/expression operand");
  }
}

template <unsigned Bits>
void DPUInstPrinter::printPCImm(const MCInst *MI, unsigned OpNo, raw_ostream &O,
                                const char *Modifier) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    O << formatHex(Op.getImm());
  } else if (Op.isExpr()) {
    Op.getExpr()->print(O, &MAI);
  } else {
    llvm_unreachable("print pc invoked with non-immediate/expression operand");
  }
}

void DPUInstPrinter::printEndianness(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O, const char *Modifier) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    switch (Op.getImm()) {
    default:
      llvm_unreachable("print endianness invoked with something else than 0/1");
    case 0:
      O << "!little";
      break;
    case 1:
      O << "!big";
      break;
    }
  } else {
    llvm_unreachable("print endianness invoked with non-immediate operand");
  }
}

void DPUInstPrinter::printCondition(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O, const char *Modifier) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    auto Cond = static_cast<DPUAsmCondition::Condition>(Op.getImm());
    O << DPUAsmCondition::toString(Cond);
  } else {
    llvm_unreachable("print condition invoked with non-immediate operand");
  }
}

void DPUInstPrinter::printImm5(const MCInst *MI, unsigned OpNo, raw_ostream &O,
                               const char *Modifier) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isImm() && "print immediate invoked with non-immediate operand");
  O << formatHex(Op.getImm());
}

void DPUInstPrinter::printImm8(const MCInst *MI, unsigned OpNo, raw_ostream &O,
                               const char *Modifier) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isImm() && "print immediate invoked with non-immediate operand");
  O << formatHex(Op.getImm());
}

void DPUInstPrinter::printImm11(const MCInst *MI, unsigned OpNo, raw_ostream &O,
                                const char *Modifier) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isImm() && "print immediate invoked with non-immediate operand");
  O << formatHex(Op.getImm());
}

void DPUInstPrinter::printImm24(const MCInst *MI, unsigned OpNo, raw_ostream &O,
                                const char *Modifier) {
  const MCOperand &Op = MI->getOperand(OpNo);
  assert(Op.isImm() && "print immediate invoked with non-immediate operand");
  O << formatHex(Op.getImm());
}

void DPUInstPrinter::printMemOperandWithImm24(const MCInst *MI,
                                              unsigned int OpNo, raw_ostream &O,
                                              const char *Modifier) {
  const MCOperand &RegOp = MI->getOperand(OpNo);
  const MCOperand &OffsetOp = MI->getOperand(OpNo + 1);
  assert(OffsetOp.isImm() && "offset to address operand is not an immediate");
  assert(RegOp.isReg() && "base to address operand is not a register");
  O << getRegisterName(RegOp.getReg()) << ", "
    << formatDec(OffsetOp.getImm());
}

void DPUInstPrinter::printPCOffsetOperand(const MCInst *MI, unsigned int OpNo,
                                          raw_ostream &O,
                                          const char *Modifier) {
  const MCOperand &OffsetOp = MI->getOperand(OpNo);
  if (OffsetOp.isExpr()) {
    const MCExpr *Expr = OffsetOp.getExpr();
    Expr->print(O, &MAI);
  } else if (OffsetOp.isImm()) {
    O << OffsetOp;
  }
}

void DPUInstPrinter::printCCOperand(const MCInst *MI, unsigned int OpNo,
                                    raw_ostream &O, const char *Modifier) {
  const MCOperand &CcOp = MI->getOperand(OpNo);
  assert(CcOp.isImm() && "CC condition is not immediate");
  ISD::CondCode IsdCondCode = (ISD::CondCode)CcOp.getImm();
  DpuBinaryCondCode BC;

  DPU::BinaryCondCode BinaryCondCode = BC.FromIsdCondCode(IsdCondCode);
  if (BinaryCondCode == DPU::COND_UNDEF_BINARY) {
    assert(false && "don't know how to handle opcode for CC operand!");
  } else {
    O << BC.AsKeyword(BinaryCondCode);
  }
}

void DPUInstPrinter::printACCOperand(const MCInst *MI, unsigned int OpNo,
                                     raw_ostream &O, const char *Modifier) {
  const MCOperand &CcOp = MI->getOperand(OpNo);
  assert(CcOp.isImm() && "CC condition is not immediate");
  ISD::CondCode IsdCondCode = (ISD::CondCode)CcOp.getImm();
  DpuUnaryCondCode UC;

  DPU::UnaryCondCode UnaryCondCode = UC.FromIsdCondCode(IsdCondCode);
  if (UnaryCondCode == DPU::COND_UNDEF_UNARY) {
    assert(false && "don't know how to handle opcode for CC operand!");
  } else {
    O << UC.AsKeyword(UnaryCondCode);
  }
}
