//===-- LanaiInstPrinter.cpp - Convert Lanai MCInst to asm syntax ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an Lanai MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "Lanai.h"
#include "LanaiInstPrinter.h"
#include "MCTargetDesc/LanaiMCExpr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

// Include the auto-generated portion of the assembly writer.
#define PRINT_ALIAS_INSTR
#include "LanaiGenAsmWriter.inc"

void LanaiInstPrinter::printRegName(raw_ostream &Ostream,
                                    unsigned RegNo) const {
  Ostream << StringRef(getRegisterName(RegNo)).lower();
}

bool LanaiInstPrinter::printInst(const MCInst *MI, raw_ostream &Ostream,
                                 StringRef Alias, unsigned OpNo0,
                                 unsigned OpNo1) {
  Ostream << "\t" << Alias << " ";
  printOperand(MI, OpNo0, Ostream);
  Ostream << ", ";
  printOperand(MI, OpNo1, Ostream);
  return true;
}

static bool usesGivenOffset(const MCInst *MI, int AddOffset) {
  unsigned AluCode = MI->getOperand(3).getImm();
  return LPAC::encodeLanaiAluCode(AluCode) == LPAC::ADD &&
         (MI->getOperand(2).getImm() == AddOffset ||
          MI->getOperand(2).getImm() == -AddOffset);
}

static bool isPreIncrementForm(const MCInst *MI, int AddOffset) {
  unsigned AluCode = MI->getOperand(3).getImm();
  return LPAC::isPreOp(AluCode) && usesGivenOffset(MI, AddOffset);
}

static bool isPostIncrementForm(const MCInst *MI, int AddOffset) {
  unsigned AluCode = MI->getOperand(3).getImm();
  return LPAC::isPostOp(AluCode) && usesGivenOffset(MI, AddOffset);
}

static StringRef decIncOperator(const MCInst *MI) {
  if (MI->getOperand(2).getImm() < 0)
    return "--";
  return "++";
}

bool LanaiInstPrinter::printMemoryLoadIncrement(const MCInst *MI,
                                                raw_ostream &Ostream,
                                                StringRef Opcode,
                                                int AddOffset) {
  if (isPreIncrementForm(MI, AddOffset)) {
    Ostream << "\t" << Opcode << "\t[" << decIncOperator(MI) << "%"
            << getRegisterName(MI->getOperand(1).getReg()) << "], %"
            << getRegisterName(MI->getOperand(0).getReg());
    return true;
  }
  if (isPostIncrementForm(MI, AddOffset)) {
    Ostream << "\t" << Opcode << "\t[%"
            << getRegisterName(MI->getOperand(1).getReg()) << decIncOperator(MI)
            << "], %" << getRegisterName(MI->getOperand(0).getReg());
    return true;
  }
  return false;
}

bool LanaiInstPrinter::printMemoryStoreIncrement(const MCInst *MI,
                                                 raw_ostream &Ostream,
                                                 StringRef Opcode,
                                                 int AddOffset) {
  if (isPreIncrementForm(MI, AddOffset)) {
    Ostream << "\t" << Opcode << "\t%"
            << getRegisterName(MI->getOperand(0).getReg()) << ", ["
            << decIncOperator(MI) << "%"
            << getRegisterName(MI->getOperand(1).getReg()) << "]";
    return true;
  }
  if (isPostIncrementForm(MI, AddOffset)) {
    Ostream << "\t" << Opcode << "\t%"
            << getRegisterName(MI->getOperand(0).getReg()) << ", [%"
            << getRegisterName(MI->getOperand(1).getReg()) << decIncOperator(MI)
            << "]";
    return true;
  }
  return false;
}

bool LanaiInstPrinter::printAlias(const MCInst *MI, raw_ostream &Ostream) {
  switch (MI->getOpcode()) {
  case Lanai::LDW_RI:
    // ld 4[*%rN], %rX => ld [++imm], %rX
    // ld -4[*%rN], %rX => ld [--imm], %rX
    // ld 4[%rN*], %rX => ld [imm++], %rX
    // ld -4[%rN*], %rX => ld [imm--], %rX
    return printMemoryLoadIncrement(MI, Ostream, "ld", 4);
  case Lanai::LDHs_RI:
    return printMemoryLoadIncrement(MI, Ostream, "ld.h", 2);
  case Lanai::LDHz_RI:
    return printMemoryLoadIncrement(MI, Ostream, "uld.h", 2);
  case Lanai::LDBs_RI:
    return printMemoryLoadIncrement(MI, Ostream, "ld.b", 1);
  case Lanai::LDBz_RI:
    return printMemoryLoadIncrement(MI, Ostream, "uld.b", 1);
  case Lanai::SW_RI:
    // st %rX, 4[*%rN] => st %rX, [++imm]
    // st %rX, -4[*%rN] => st %rX, [--imm]
    // st %rX, 4[%rN*] => st %rX, [imm++]
    // st %rX, -4[%rN*] => st %rX, [imm--]
    return printMemoryStoreIncrement(MI, Ostream, "st", 4);
  case Lanai::STH_RI:
    return printMemoryStoreIncrement(MI, Ostream, "st.h", 2);
  case Lanai::STB_RI:
    return printMemoryStoreIncrement(MI, Ostream, "st.b", 1);
  default:
    return false;
  }
}

void LanaiInstPrinter::printInst(const MCInst *MI, raw_ostream &Ostream,
                                 StringRef Annotation,
                                 const MCSubtargetInfo &STI) {
  if (!printAlias(MI, Ostream) && !printAliasInstr(MI, Ostream))
    printInstruction(MI, Ostream);
  printAnnotation(Ostream, Annotation);
}

static void printExpr(const MCAsmInfo &MAI, const MCExpr &Expr,
                      raw_ostream &Ostream) {
  const MCExpr *SRE;

  if (const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(&Expr))
    SRE = dyn_cast<LanaiMCExpr>(BE->getLHS());
  else if (isa<LanaiMCExpr>(&Expr)) {
    SRE = dyn_cast<LanaiMCExpr>(&Expr);
  } else {
    SRE = dyn_cast<MCSymbolRefExpr>(&Expr);
  }
  assert(SRE && "Unexpected MCExpr type.");

  SRE->print(Ostream, &MAI);
}

void LanaiInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &Ostream,
                                    const char *Modifier) {
  assert((Modifier == 0 || Modifier[0] == 0) && "No modifiers supported");
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg())
    Ostream << "%" << getRegisterName(Op.getReg());
  else if (Op.isImm())
    Ostream << formatHex(Op.getImm());
  else {
    assert(Op.isExpr() && "Expected an expression");
    printExpr(MAI, *Op.getExpr(), Ostream);
  }
}

void LanaiInstPrinter::printMemImmOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &Ostream) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    Ostream << '[' << formatHex(Op.getImm()) << ']';
  } else {
    // Symbolic operand will be lowered to immediate value by linker
    assert(Op.isExpr() && "Expected an expression");
    Ostream << '[';
    printExpr(MAI, *Op.getExpr(), Ostream);
    Ostream << ']';
  }
}

void LanaiInstPrinter::printHi16ImmOperand(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &Ostream) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    Ostream << formatHex(Op.getImm() << 16);
  } else {
    // Symbolic operand will be lowered to immediate value by linker
    assert(Op.isExpr() && "Expected an expression");
    printExpr(MAI, *Op.getExpr(), Ostream);
  }
}

void LanaiInstPrinter::printHi16AndImmOperand(const MCInst *MI, unsigned OpNo,
                                              raw_ostream &Ostream) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    Ostream << formatHex((Op.getImm() << 16) | 0xffff);
  } else {
    // Symbolic operand will be lowered to immediate value by linker
    assert(Op.isExpr() && "Expected an expression");
    printExpr(MAI, *Op.getExpr(), Ostream);
  }
}

void LanaiInstPrinter::printLo16AndImmOperand(const MCInst *MI, unsigned OpNo,
                                              raw_ostream &Ostream) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    Ostream << formatHex(0xffff0000 | Op.getImm());
  } else {
    // Symbolic operand will be lowered to immediate value by linker
    assert(Op.isExpr() && "Expected an expression");
    printExpr(MAI, *Op.getExpr(), Ostream);
  }
}

static void printMemoryBaseRegister(raw_ostream &Ostream, const unsigned AluCode,
                             const MCOperand &RegOp) {
  assert(RegOp.isReg() && "Register operand expected");
  Ostream << "[";
  if (LPAC::isPreOp(AluCode))
    Ostream << "*";
  Ostream << "%" << LanaiInstPrinter::getRegisterName(RegOp.getReg());
  if (LPAC::isPostOp(AluCode))
    Ostream << "*";
  Ostream << "]";
}

template <unsigned SizeInBits>
static void printMemoryImmediateOffset(const MCAsmInfo &MAI,
                                       const MCOperand &OffsetOp,
                                       raw_ostream &Ostream) {
  assert((OffsetOp.isImm() || OffsetOp.isExpr()) && "Immediate expected");
  if (OffsetOp.isImm()) {
    assert(isInt<SizeInBits>(OffsetOp.getImm()) && "Constant value truncated");
    Ostream << OffsetOp.getImm();
  } else
    printExpr(MAI, *OffsetOp.getExpr(), Ostream);
}

void LanaiInstPrinter::printMemRiOperand(const MCInst *MI, int OpNo,
                                         raw_ostream &Ostream,
                                         const char *Modifier) {
  const MCOperand &RegOp = MI->getOperand(OpNo);
  const MCOperand &OffsetOp = MI->getOperand(OpNo + 1);
  const MCOperand &AluOp = MI->getOperand(OpNo + 2);
  const unsigned AluCode = AluOp.getImm();

  // Offset
  printMemoryImmediateOffset<16>(MAI, OffsetOp, Ostream);

  // Register
  printMemoryBaseRegister(Ostream, AluCode, RegOp);
}

void LanaiInstPrinter::printMemRrOperand(const MCInst *MI, int OpNo,
                                         raw_ostream &Ostream,
                                         const char *Modifier) {
  const MCOperand &RegOp = MI->getOperand(OpNo);
  const MCOperand &OffsetOp = MI->getOperand(OpNo + 1);
  const MCOperand &AluOp = MI->getOperand(OpNo + 2);
  const unsigned AluCode = AluOp.getImm();
  assert(OffsetOp.isReg() && RegOp.isReg() && "Registers expected.");

  // [ Base OP Offset ]
  Ostream << "[";
  if (LPAC::isPreOp(AluCode))
    Ostream << "*";
  Ostream << "%" << getRegisterName(RegOp.getReg());
  if (LPAC::isPostOp(AluCode))
    Ostream << "*";
  Ostream << " " << LPAC::lanaiAluCodeToString(AluCode) << " ";
  Ostream << "%" << getRegisterName(OffsetOp.getReg());
  Ostream << "]";
}

void LanaiInstPrinter::printMemSplsOperand(const MCInst *MI, int OpNo,
                                           raw_ostream &Ostream,
                                           const char *Modifier) {
  const MCOperand &RegOp = MI->getOperand(OpNo);
  const MCOperand &OffsetOp = MI->getOperand(OpNo + 1);
  const MCOperand &AluOp = MI->getOperand(OpNo + 2);
  const unsigned AluCode = AluOp.getImm();

  // Offset
  printMemoryImmediateOffset<10>(MAI, OffsetOp, Ostream);

  // Register
  printMemoryBaseRegister(Ostream, AluCode, RegOp);
}

void LanaiInstPrinter::printCCOperand(const MCInst *MI, int OpNo,
                                      raw_ostream &Ostream) {
  const int CC = static_cast<const int>(MI->getOperand(OpNo).getImm());
  Ostream << lanaiCondCodeToString(static_cast<LPCC::CondCode>(CC));
}
