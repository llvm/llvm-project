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

#define DEBUG_TYPE "asm-printer"
#include "PPCInstPrinter.h"
#include "MCTargetDesc/PPCBaseInfo.h"
#include "MCTargetDesc/PPCPredicates.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

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
  
  printInstruction(MI, O);
  printAnnotation(O, Annot);
}


void PPCInstPrinter::printPredicateOperand(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O, 
                                           const char *Modifier) {
  unsigned Code = MI->getOperand(OpNo).getImm();
  if (!Modifier) {
    unsigned CCReg = MI->getOperand(OpNo+1).getReg();
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

    // Print the CR bit number. The Code is ((BI << 5) | BO) for a
    // BCC, but we must have the positive form here (BO == 12)
    unsigned BI = Code >> 5;
    assert((Code & 0xF) == 12 &&
           "BO in predicate bit must have the positive form");

    unsigned Value = 4*RegNo + BI;
    O << Value;
    return;
  }

  if (StringRef(Modifier) == "cc") {
    switch ((PPC::Predicate)Code) {
    case PPC::PRED_ALWAYS: return; // Don't print anything for always.
    case PPC::PRED_LT: O << "lt"; return;
    case PPC::PRED_LE: O << "le"; return;
    case PPC::PRED_EQ: O << "eq"; return;
    case PPC::PRED_GE: O << "ge"; return;
    case PPC::PRED_GT: O << "gt"; return;
    case PPC::PRED_NE: O << "ne"; return;
    case PPC::PRED_UN: O << "un"; return;
    case PPC::PRED_NU: O << "nu"; return;
    }
  }
  
  assert(StringRef(Modifier) == "reg" &&
         "Need to specify 'cc' or 'reg' as predicate op modifier!");
  // Don't print the register for 'always'.
  if (Code == PPC::PRED_ALWAYS) return;
  printOperand(MI, OpNo+1, O);
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
  O << (short)MI->getOperand(OpNo).getImm();
}

void PPCInstPrinter::printU16ImmOperand(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  O << (unsigned short)MI->getOperand(OpNo).getImm();
}

void PPCInstPrinter::printS16X4ImmOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  if (MI->getOperand(OpNo).isImm())
    O << (short)(MI->getOperand(OpNo).getImm()*4);
  else
    printOperand(MI, OpNo, O);
}

void PPCInstPrinter::printBranchOperand(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  if (!MI->getOperand(OpNo).isImm())
    return printOperand(MI, OpNo, O);

  // Branches can take an immediate operand.  This is used by the branch
  // selection pass to print $+8, an eight byte displacement from the PC.
  O << "$+";
  printAbsAddrOperand(MI, OpNo, O);
}

void PPCInstPrinter::printAbsAddrOperand(const MCInst *MI, unsigned OpNo,
                                         raw_ostream &O) {
  O << (int)MI->getOperand(OpNo).getImm()*4;
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
  printSymbolLo(MI, OpNo, O);
  O << '(';
  if (MI->getOperand(OpNo+1).getReg() == PPC::R0)
    O << "0";
  else
    printOperand(MI, OpNo+1, O);
  O << ')';
}

void PPCInstPrinter::printMemRegImmShifted(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  if (MI->getOperand(OpNo).isImm())
    printS16X4ImmOperand(MI, OpNo, O);
  else
    printSymbolLo(MI, OpNo, O);
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



/// stripRegisterPrefix - This method strips the character prefix from a
/// register name so that only the number is left.  Used by for linux asm.
static const char *stripRegisterPrefix(const char *RegName) {
  switch (RegName[0]) {
  case 'r':
  case 'f':
  case 'v': return RegName + 1;
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
  
void PPCInstPrinter::printSymbolLo(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  if (MI->getOperand(OpNo).isImm())
    return printS16ImmOperand(MI, OpNo, O);
  
  // FIXME: This is a terrible hack because we can't encode lo16() as an operand
  // flag of a subtraction.  See the FIXME in GetSymbolRef in PPCMCInstLower.
  if (MI->getOperand(OpNo).isExpr() &&
      isa<MCBinaryExpr>(MI->getOperand(OpNo).getExpr())) {
    O << "lo16(";
    printOperand(MI, OpNo, O);
    O << ')';
  } else {
    printOperand(MI, OpNo, O);
  }
}

void PPCInstPrinter::printSymbolHi(const MCInst *MI, unsigned OpNo,
                                   raw_ostream &O) {
  if (MI->getOperand(OpNo).isImm())
    return printS16ImmOperand(MI, OpNo, O);

  // FIXME: This is a terrible hack because we can't encode lo16() as an operand
  // flag of a subtraction.  See the FIXME in GetSymbolRef in PPCMCInstLower.
  if (MI->getOperand(OpNo).isExpr() &&
      isa<MCBinaryExpr>(MI->getOperand(OpNo).getExpr())) {
    O << "ha16(";
    printOperand(MI, OpNo, O);
    O << ')';
  } else {
    printOperand(MI, OpNo, O);
  }
}


