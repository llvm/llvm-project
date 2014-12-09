//===-- PPCInstPrinter.cpp - Convert PPC MCInst to assembly syntax --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an PPC MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "PPCInstPrinter.h"
#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "MCTargetDesc/PPCPredicates.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOpcodes.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

// FIXME: Once the integrated assembler supports full register names, tie this
// to the verbose-asm setting.
static cl::opt<bool>
FullRegNames("ppc-asm-full-reg-names", cl::Hidden, cl::init(false),
             cl::desc("Use full register names when printing assembly"));

#include "PPCGenAsmWriter.inc"

void PPCInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  OS << getRegisterName(RegNo);
}

void PPCInstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                               StringRef Annot) {
  // Check for slwi/srwi mnemonics.
  if (MI->getOpcode() == PPC::RLWINM) {
    unsigned char SH = MI->getOperand(2).getImm();
    unsigned char MB = MI->getOperand(3).getImm();
    unsigned char ME = MI->getOperand(4).getImm();
    bool useSubstituteMnemonic = false;
    if (SH <= 31 && MB == 0 && ME == (31-SH)) {
      O << "\tslwi "; useSubstituteMnemonic = true;
    }
    if (SH <= 31 && MB == (32-SH) && ME == 31) {
      O << "\tsrwi "; useSubstituteMnemonic = true;
      SH = 32-SH;
    }
    if (useSubstituteMnemonic) {
      printOperand(MI, 0, O);
      O << ", ";
      printOperand(MI, 1, O);
      O << ", " << (unsigned int)SH;

      printAnnotation(O, Annot);
      return;
    }
  }
  
  if ((MI->getOpcode() == PPC::OR || MI->getOpcode() == PPC::OR8) &&
      MI->getOperand(1).getReg() == MI->getOperand(2).getReg()) {
    O << "\tmr ";
    printOperand(MI, 0, O);
    O << ", ";
    printOperand(MI, 1, O);
    printAnnotation(O, Annot);
    return;
  }
  
  if (MI->getOpcode() == PPC::RLDICR) {
    unsigned char SH = MI->getOperand(2).getImm();
    unsigned char ME = MI->getOperand(3).getImm();
    // rldicr RA, RS, SH, 63-SH == sldi RA, RS, SH
    if (63-SH == ME) {
      O << "\tsldi ";
      printOperand(MI, 0, O);
      O << ", ";
      printOperand(MI, 1, O);
      O << ", " << (unsigned int)SH;
      printAnnotation(O, Annot);
      return;
    }
  }
  
  // For fast-isel, a COPY_TO_REGCLASS may survive this long.  This is
  // used when converting a 32-bit float to a 64-bit float as part of
  // conversion to an integer (see PPCFastISel.cpp:SelectFPToI()),
  // as otherwise we have problems with incorrect register classes
  // in machine instruction verification.  For now, just avoid trying
  // to print it as such an instruction has no effect (a 32-bit float
  // in a register is already in 64-bit form, just with lower
  // precision).  FIXME: Is there a better solution?
  if (MI->getOpcode() == TargetOpcode::COPY_TO_REGCLASS)
    return;
  
  printInstruction(MI, O);
  printAnnotation(O, Annot);
}


void PPCInstPrinter::printPredicateOperand(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O, 
                                           const char *Modifier) {
  unsigned Code = MI->getOperand(OpNo).getImm();

  if (StringRef(Modifier) == "cc") {
    switch ((PPC::Predicate)Code) {
    case PPC::PRED_LT_MINUS:
    case PPC::PRED_LT_PLUS:
    case PPC::PRED_LT:
      O << "lt";
      return;
    case PPC::PRED_LE_MINUS:
    case PPC::PRED_LE_PLUS:
    case PPC::PRED_LE:
      O << "le";
      return;
    case PPC::PRED_EQ_MINUS:
    case PPC::PRED_EQ_PLUS:
    case PPC::PRED_EQ:
      O << "eq";
      return;
    case PPC::PRED_GE_MINUS:
    case PPC::PRED_GE_PLUS:
    case PPC::PRED_GE:
      O << "ge";
      return;
    case PPC::PRED_GT_MINUS:
    case PPC::PRED_GT_PLUS:
    case PPC::PRED_GT:
      O << "gt";
      return;
    case PPC::PRED_NE_MINUS:
    case PPC::PRED_NE_PLUS:
    case PPC::PRED_NE:
      O << "ne";
      return;
    case PPC::PRED_UN_MINUS:
    case PPC::PRED_UN_PLUS:
    case PPC::PRED_UN:
      O << "un";
      return;
    case PPC::PRED_NU_MINUS:
    case PPC::PRED_NU_PLUS:
    case PPC::PRED_NU:
      O << "nu";
      return;
    case PPC::PRED_BIT_SET:
    case PPC::PRED_BIT_UNSET:
      llvm_unreachable("Invalid use of bit predicate code");
    }
    llvm_unreachable("Invalid predicate code");
  }

  if (StringRef(Modifier) == "pm") {
    switch ((PPC::Predicate)Code) {
    case PPC::PRED_LT:
    case PPC::PRED_LE:
    case PPC::PRED_EQ:
    case PPC::PRED_GE:
    case PPC::PRED_GT:
    case PPC::PRED_NE:
    case PPC::PRED_UN:
    case PPC::PRED_NU:
      return;
    case PPC::PRED_LT_MINUS:
    case PPC::PRED_LE_MINUS:
    case PPC::PRED_EQ_MINUS:
    case PPC::PRED_GE_MINUS:
    case PPC::PRED_GT_MINUS:
    case PPC::PRED_NE_MINUS:
    case PPC::PRED_UN_MINUS:
    case PPC::PRED_NU_MINUS:
      O << "-";
      return;
    case PPC::PRED_LT_PLUS:
    case PPC::PRED_LE_PLUS:
    case PPC::PRED_EQ_PLUS:
    case PPC::PRED_GE_PLUS:
    case PPC::PRED_GT_PLUS:
    case PPC::PRED_NE_PLUS:
    case PPC::PRED_UN_PLUS:
    case PPC::PRED_NU_PLUS:
      O << "+";
      return;
    case PPC::PRED_BIT_SET:
    case PPC::PRED_BIT_UNSET:
      llvm_unreachable("Invalid use of bit predicate code");
    }
    llvm_unreachable("Invalid predicate code");
  }
  
  assert(StringRef(Modifier) == "reg" &&
         "Need to specify 'cc', 'pm' or 'reg' as predicate op modifier!");
  printOperand(MI, OpNo+1, O);
}

void PPCInstPrinter::printU2ImmOperand(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  unsigned int Value = MI->getOperand(OpNo).getImm();
  assert(Value <= 3 && "Invalid u2imm argument!");
  O << (unsigned int)Value;
}

void PPCInstPrinter::printS5ImmOperand(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  int Value = MI->getOperand(OpNo).getImm();
  Value = SignExtend32<5>(Value);
  O << (int)Value;
}

void PPCInstPrinter::printU5ImmOperand(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  unsigned int Value = MI->getOperand(OpNo).getImm();
  assert(Value <= 31 && "Invalid u5imm argument!");
  O << (unsigned int)Value;
}

void PPCInstPrinter::printU6ImmOperand(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  unsigned int Value = MI->getOperand(OpNo).getImm();
  assert(Value <= 63 && "Invalid u6imm argument!");
  O << (unsigned int)Value;
}

void PPCInstPrinter::printS16ImmOperand(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  if (MI->getOperand(OpNo).isImm())
    O << (short)MI->getOperand(OpNo).getImm();
  else
    printOperand(MI, OpNo, O);
}

void PPCInstPrinter::printU16ImmOperand(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  if (MI->getOperand(OpNo).isImm())
    O << (unsigned short)MI->getOperand(OpNo).getImm();
  else
    printOperand(MI, OpNo, O);
}

void PPCInstPrinter::printBranchOperand(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  if (!MI->getOperand(OpNo).isImm())
    return printOperand(MI, OpNo, O);

  // Branches can take an immediate operand.  This is used by the branch
  // selection pass to print .+8, an eight byte displacement from the PC.
  O << ".+";
  printAbsBranchOperand(MI, OpNo, O);
}

void PPCInstPrinter::printAbsBranchOperand(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  if (!MI->getOperand(OpNo).isImm())
    return printOperand(MI, OpNo, O);

  O << SignExtend32<32>((unsigned)MI->getOperand(OpNo).getImm() << 2);
}


void PPCInstPrinter::printcrbitm(const MCInst *MI, unsigned OpNo,
                                 raw_ostream &O) {
  unsigned CCReg = MI->getOperand(OpNo).getReg();
  unsigned RegNo;
  switch (CCReg) {
  default: llvm_unreachable("Unknown CR register");
  case PPC::CR0: RegNo = 0; break;
  case PPC::CR1: RegNo = 1; break;
  case PPC::CR2: RegNo = 2; break;
  case PPC::CR3: RegNo = 3; break;
  case PPC::CR4: RegNo = 4; break;
  case PPC::CR5: RegNo = 5; break;
  case PPC::CR6: RegNo = 6; break;
  case PPC::CR7: RegNo = 7; break;
  }
  O << (0x80 >> RegNo);
}

void PPCInstPrinter::printMemRegImm(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O) {
  printS16ImmOperand(MI, OpNo, O);
  O << '(';
  if (MI->getOperand(OpNo+1).getReg() == PPC::R0)
    O << "0";
  else
    printOperand(MI, OpNo+1, O);
  O << ')';
}

void PPCInstPrinter::printMemRegReg(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O) {
  // When used as the base register, r0 reads constant zero rather than
  // the value contained in the register.  For this reason, the darwin
  // assembler requires that we print r0 as 0 (no r) when used as the base.
  if (MI->getOperand(OpNo).getReg() == PPC::R0)
    O << "0";
  else
    printOperand(MI, OpNo, O);
  O << ", ";
  printOperand(MI, OpNo+1, O);
}

void PPCInstPrinter::printTLSCall(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
  // On PPC64, VariantKind is VK_None, but on PPC32, it's VK_PLT, and it must
  // come at the _end_ of the expression.
  const MCOperand &Op = MI->getOperand(OpNo);
  const MCSymbolRefExpr &refExp = cast<MCSymbolRefExpr>(*Op.getExpr());
  O << refExp.getSymbol().getName();
  O << '(';
  printOperand(MI, OpNo+1, O);
  O << ')';
  if (refExp.getKind() != MCSymbolRefExpr::VK_None)
    O << '@' << MCSymbolRefExpr::getVariantKindName(refExp.getKind());
}


/// stripRegisterPrefix - This method strips the character prefix from a
/// register name so that only the number is left.  Used by for linux asm.
static const char *stripRegisterPrefix(const char *RegName) {
  if (FullRegNames)
    return RegName;

  switch (RegName[0]) {
  case 'r':
  case 'f':
  case 'v':
    if (RegName[1] == 's')
      return RegName + 2;
    return RegName + 1;
  case 'c': if (RegName[1] == 'r') return RegName + 2;
  }
  
  return RegName;
}

void PPCInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    const char *RegName = getRegisterName(Op.getReg());
    // The linux and AIX assembler does not take register prefixes.
    if (!isDarwinSyntax())
      RegName = stripRegisterPrefix(RegName);
    
    O << RegName;
    return;
  }
  
  if (Op.isImm()) {
    O << Op.getImm();
    return;
  }
  
  assert(Op.isExpr() && "unknown operand kind in printOperand");
  O << *Op.getExpr();
}

