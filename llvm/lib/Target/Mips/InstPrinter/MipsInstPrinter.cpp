//===-- MipsInstPrinter.cpp - Convert Mips MCInst to assembly syntax ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an Mips MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "MipsInstPrinter.h"
#include "MipsInstrInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#include "MipsGenAsmWriter.inc"

const char* Mips::MipsFCCToString(Mips::CondCode CC) {
  switch (CC) {
  case FCOND_F:
  case FCOND_T:   return "f";
  case FCOND_UN:
  case FCOND_OR:  return "un";
  case FCOND_OEQ:
  case FCOND_UNE: return "eq";
  case FCOND_UEQ:
  case FCOND_ONE: return "ueq";
  case FCOND_OLT:
  case FCOND_UGE: return "olt";
  case FCOND_ULT:
  case FCOND_OGE: return "ult";
  case FCOND_OLE:
  case FCOND_UGT: return "ole";
  case FCOND_ULE:
  case FCOND_OGT: return "ule";
  case FCOND_SF:
  case FCOND_ST:  return "sf";
  case FCOND_NGLE:
  case FCOND_GLE: return "ngle";
  case FCOND_SEQ:
  case FCOND_SNE: return "seq";
  case FCOND_NGL:
  case FCOND_GL:  return "ngl";
  case FCOND_LT:
  case FCOND_NLT: return "lt";
  case FCOND_NGE:
  case FCOND_GE:  return "nge";
  case FCOND_LE:
  case FCOND_NLE: return "le";
  case FCOND_NGT:
  case FCOND_GT:  return "ngt";
  }
  llvm_unreachable("Impossible condition code!");
}

void MipsInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  OS << '$' << StringRef(getRegisterName(RegNo)).lower();
}

void MipsInstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                                StringRef Annot) {
  switch (MI->getOpcode()) {
  default:
    break;
  case Mips::RDHWR:
  case Mips::RDHWR64:
    O << "\t.set\tpush\n";
    O << "\t.set\tmips32r2\n";
  }

  printInstruction(MI, O);
  printAnnotation(O, Annot);

  switch (MI->getOpcode()) {
  default:
    break;
  case Mips::RDHWR:
  case Mips::RDHWR64:
    O << "\n\t.set\tpop";
  }
}

static void printExpr(const MCExpr *Expr, raw_ostream &OS) {
  int Offset = 0;
  const MCSymbolRefExpr *SRE;

  if (const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(Expr)) {
    SRE = dyn_cast<MCSymbolRefExpr>(BE->getLHS());
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(BE->getRHS());
    assert(SRE && CE && "Binary expression must be sym+const.");
    Offset = CE->getValue();
  }
  else if (!(SRE = dyn_cast<MCSymbolRefExpr>(Expr)))
    assert(false && "Unexpected MCExpr type.");

  MCSymbolRefExpr::VariantKind Kind = SRE->getKind();

  switch (Kind) {
  default:                                 llvm_unreachable("Invalid kind!");
  case MCSymbolRefExpr::VK_None:           break;
  case MCSymbolRefExpr::VK_Mips_GPREL:     OS << "%gp_rel("; break;
  case MCSymbolRefExpr::VK_Mips_GOT_CALL:  OS << "%call16("; break;
  case MCSymbolRefExpr::VK_Mips_GOT16:     OS << "%got(";    break;
  case MCSymbolRefExpr::VK_Mips_GOT:       OS << "%got(";    break;
  case MCSymbolRefExpr::VK_Mips_ABS_HI:    OS << "%hi(";     break;
  case MCSymbolRefExpr::VK_Mips_ABS_LO:    OS << "%lo(";     break;
  case MCSymbolRefExpr::VK_Mips_TLSGD:     OS << "%tlsgd(";  break;
  case MCSymbolRefExpr::VK_Mips_TLSLDM:    OS << "%tlsldm(";  break;
  case MCSymbolRefExpr::VK_Mips_DTPREL_HI: OS << "%dtprel_hi(";  break;
  case MCSymbolRefExpr::VK_Mips_DTPREL_LO: OS << "%dtprel_lo(";  break;
  case MCSymbolRefExpr::VK_Mips_GOTTPREL:  OS << "%gottprel("; break;
  case MCSymbolRefExpr::VK_Mips_TPREL_HI:  OS << "%tprel_hi("; break;
  case MCSymbolRefExpr::VK_Mips_TPREL_LO:  OS << "%tprel_lo("; break;
  case MCSymbolRefExpr::VK_Mips_GPOFF_HI:  OS << "%hi(%neg(%gp_rel("; break;
  case MCSymbolRefExpr::VK_Mips_GPOFF_LO:  OS << "%lo(%neg(%gp_rel("; break;
  case MCSymbolRefExpr::VK_Mips_GOT_DISP:  OS << "%got_disp("; break;
  case MCSymbolRefExpr::VK_Mips_GOT_PAGE:  OS << "%got_page("; break;
  case MCSymbolRefExpr::VK_Mips_GOT_OFST:  OS << "%got_ofst("; break;
  case MCSymbolRefExpr::VK_Mips_HIGHER:    OS << "%higher("; break;
  case MCSymbolRefExpr::VK_Mips_HIGHEST:   OS << "%highest("; break;
  case MCSymbolRefExpr::VK_Mips_GOT_HI16:  OS << "%got_hi("; break;
  case MCSymbolRefExpr::VK_Mips_GOT_LO16:  OS << "%got_lo("; break;
  case MCSymbolRefExpr::VK_Mips_CALL_HI16: OS << "%call_hi("; break;
  case MCSymbolRefExpr::VK_Mips_CALL_LO16: OS << "%call_lo("; break;
  }

  OS << SRE->getSymbol();

  if (Offset) {
    if (Offset > 0)
      OS << '+';
    OS << Offset;
  }

  if ((Kind == MCSymbolRefExpr::VK_Mips_GPOFF_HI) ||
      (Kind == MCSymbolRefExpr::VK_Mips_GPOFF_LO))
    OS << ")))";
  else if (Kind != MCSymbolRefExpr::VK_None)
    OS << ')';
}

void MipsInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
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
  printExpr(Op.getExpr(), O);
}

void MipsInstPrinter::printUnsignedImm(const MCInst *MI, int opNum,
                                       raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(opNum);
  if (MO.isImm())
    O << (unsigned short int)MO.getImm();
  else
    printOperand(MI, opNum, O);
}

void MipsInstPrinter::
printMemOperand(const MCInst *MI, int opNum, raw_ostream &O) {
  // Load/Store memory operands -- imm($reg)
  // If PIC target the target is loaded as the
  // pattern lw $25,%call16($28)
  printOperand(MI, opNum+1, O);
  O << "(";
  printOperand(MI, opNum, O);
  O << ")";
}

void MipsInstPrinter::
printMemOperandEA(const MCInst *MI, int opNum, raw_ostream &O) {
  // when using stack locations for not load/store instructions
  // print the same way as all normal 3 operand instructions.
  printOperand(MI, opNum, O);
  O << ", ";
  printOperand(MI, opNum+1, O);
  return;
}

void MipsInstPrinter::
printFCCOperand(const MCInst *MI, int opNum, raw_ostream &O) {
  const MCOperand& MO = MI->getOperand(opNum);
  O << MipsFCCToString((Mips::CondCode)MO.getImm());
}
