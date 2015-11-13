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

#include "MipsInstPrinter.h"
#include "MCTargetDesc/MipsMCExpr.h"
#include "MipsInstrInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#define PRINT_ALIAS_INSTR
#include "MipsGenAsmWriter.inc"

template<unsigned R>
static bool isReg(const MCInst &MI, unsigned OpNo) {
  assert(MI.getOperand(OpNo).isReg() && "Register operand expected.");
  return MI.getOperand(OpNo).getReg() == R;
}

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
                                StringRef Annot, const MCSubtargetInfo &STI) {
  switch (MI->getOpcode()) {
  default:
    break;
  case Mips::RDHWR:
  case Mips::RDHWR64:
    O << "\t.set\tpush\n";
    O << "\t.set\tmips32r2\n";
    break;
  case Mips::Save16:
    O << "\tsave\t";
    printSaveRestore(MI, O);
    O << " # 16 bit inst\n";
    return;
  case Mips::SaveX16:
    O << "\tsave\t";
    printSaveRestore(MI, O);
    O << "\n";
    return;
  case Mips::Restore16:
    O << "\trestore\t";
    printSaveRestore(MI, O);
    O << " # 16 bit inst\n";
    return;
  case Mips::RestoreX16:
    O << "\trestore\t";
    printSaveRestore(MI, O);
    O << "\n";
    return;
  }

  // Try to print any aliases first.
  if (!printAliasInstr(MI, O) && !printAlias(*MI, O))
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

static void printExpr(const MCExpr *Expr, const MCAsmInfo *MAI,
                      raw_ostream &OS) {
  int Offset = 0;
  const MCSymbolRefExpr *SRE;

  if (const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(Expr)) {
    SRE = dyn_cast<MCSymbolRefExpr>(BE->getLHS());
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(BE->getRHS());
    assert(SRE && CE && "Binary expression must be sym+const.");
    Offset = CE->getValue();
  } else if (const MipsMCExpr *ME = dyn_cast<MipsMCExpr>(Expr)) {
    ME->print(OS, MAI);
    return;
  } else
    SRE = cast<MCSymbolRefExpr>(Expr);

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
  case MCSymbolRefExpr::VK_Mips_PCREL_HI16: OS << "%pcrel_hi("; break;
  case MCSymbolRefExpr::VK_Mips_PCREL_LO16: OS << "%pcrel_lo("; break;
  }

  SRE->getSymbol().print(OS, MAI);

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
  printExpr(Op.getExpr(), &MAI, O);
}

void MipsInstPrinter::printUnsignedImm(const MCInst *MI, int opNum,
                                       raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(opNum);
  if (MO.isImm())
    O << (unsigned short int)MO.getImm();
  else
    printOperand(MI, opNum, O);
}

void MipsInstPrinter::printUnsignedImm8(const MCInst *MI, int opNum,
                                        raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(opNum);
  if (MO.isImm())
    O << (unsigned short int)(unsigned char)MO.getImm();
  else
    printOperand(MI, opNum, O);
}

void MipsInstPrinter::
printMemOperand(const MCInst *MI, int opNum, raw_ostream &O) {
  // Load/Store memory operands -- imm($reg)
  // If PIC target the target is loaded as the
  // pattern lw $25,%call16($28)

  // opNum can be invalid if instruction had reglist as operand.
  // MemOperand is always last operand of instruction (base + offset).
  switch (MI->getOpcode()) {
  default:
    break;
  case Mips::SWM32_MM:
  case Mips::LWM32_MM:
  case Mips::SWM16_MM:
  case Mips::SWM16_MMR6:
  case Mips::LWM16_MM:
  case Mips::LWM16_MMR6:
    opNum = MI->getNumOperands() - 2;
    break;
  }

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

void MipsInstPrinter::
printRegisterPair(const MCInst *MI, int opNum, raw_ostream &O) {
  printRegName(O, MI->getOperand(opNum).getReg());
}

void MipsInstPrinter::
printSHFMask(const MCInst *MI, int opNum, raw_ostream &O) {
  llvm_unreachable("TODO");
}

bool MipsInstPrinter::printAlias(const char *Str, const MCInst &MI,
                                 unsigned OpNo, raw_ostream &OS) {
  OS << "\t" << Str << "\t";
  printOperand(&MI, OpNo, OS);
  return true;
}

bool MipsInstPrinter::printAlias(const char *Str, const MCInst &MI,
                                 unsigned OpNo0, unsigned OpNo1,
                                 raw_ostream &OS) {
  printAlias(Str, MI, OpNo0, OS);
  OS << ", ";
  printOperand(&MI, OpNo1, OS);
  return true;
}

bool MipsInstPrinter::printAlias(const MCInst &MI, raw_ostream &OS) {
  switch (MI.getOpcode()) {
  case Mips::BEQ:
  case Mips::BEQ_MM:
    // beq $zero, $zero, $L2 => b $L2
    // beq $r0, $zero, $L2 => beqz $r0, $L2
    return (isReg<Mips::ZERO>(MI, 0) && isReg<Mips::ZERO>(MI, 1) &&
            printAlias("b", MI, 2, OS)) ||
           (isReg<Mips::ZERO>(MI, 1) && printAlias("beqz", MI, 0, 2, OS));
  case Mips::BEQ64:
    // beq $r0, $zero, $L2 => beqz $r0, $L2
    return isReg<Mips::ZERO_64>(MI, 1) && printAlias("beqz", MI, 0, 2, OS);
  case Mips::BNE:
    // bne $r0, $zero, $L2 => bnez $r0, $L2
    return isReg<Mips::ZERO>(MI, 1) && printAlias("bnez", MI, 0, 2, OS);
  case Mips::BNE64:
    // bne $r0, $zero, $L2 => bnez $r0, $L2
    return isReg<Mips::ZERO_64>(MI, 1) && printAlias("bnez", MI, 0, 2, OS);
  case Mips::BGEZAL:
    // bgezal $zero, $L1 => bal $L1
    return isReg<Mips::ZERO>(MI, 0) && printAlias("bal", MI, 1, OS);
  case Mips::BC1T:
    // bc1t $fcc0, $L1 => bc1t $L1
    return isReg<Mips::FCC0>(MI, 0) && printAlias("bc1t", MI, 1, OS);
  case Mips::BC1F:
    // bc1f $fcc0, $L1 => bc1f $L1
    return isReg<Mips::FCC0>(MI, 0) && printAlias("bc1f", MI, 1, OS);
  case Mips::JALR:
    // jalr $ra, $r1 => jalr $r1
    return isReg<Mips::RA>(MI, 0) && printAlias("jalr", MI, 1, OS);
  case Mips::JALR64:
    // jalr $ra, $r1 => jalr $r1
    return isReg<Mips::RA_64>(MI, 0) && printAlias("jalr", MI, 1, OS);
  case Mips::NOR:
  case Mips::NOR_MM:
    // nor $r0, $r1, $zero => not $r0, $r1
    return isReg<Mips::ZERO>(MI, 2) && printAlias("not", MI, 0, 1, OS);
  case Mips::NOR64:
    // nor $r0, $r1, $zero => not $r0, $r1
    return isReg<Mips::ZERO_64>(MI, 2) && printAlias("not", MI, 0, 1, OS);
  case Mips::OR:
    // or $r0, $r1, $zero => move $r0, $r1
    return isReg<Mips::ZERO>(MI, 2) && printAlias("move", MI, 0, 1, OS);
  default: return false;
  }
}

void MipsInstPrinter::printSaveRestore(const MCInst *MI, raw_ostream &O) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    if (i != 0) O << ", ";
    if (MI->getOperand(i).isReg())
      printRegName(O, MI->getOperand(i).getReg());
    else
      printUnsignedImm(MI, i, O);
  }
}

void MipsInstPrinter::
printRegisterList(const MCInst *MI, int opNum, raw_ostream &O) {
  // - 2 because register List is always first operand of instruction and it is
  // always followed by memory operand (base + offset).
  for (int i = opNum, e = MI->getNumOperands() - 2; i != e; ++i) {
    if (i != opNum)
      O << ", ";
    printRegName(O, MI->getOperand(i).getReg());
  }
}
