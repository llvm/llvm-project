//===-- ARMInstPrinter.cpp - Convert ARM MCInst to assembly syntax --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an ARM MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "ARMInstPrinter.h"
#include "MCTargetDesc/ARMBaseInfo.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#include "ARMGenAsmWriter.inc"

/// translateShiftImm - Convert shift immediate from 0-31 to 1-32 for printing.
///
/// getSORegOffset returns an integer from 0-31, representing '32' as 0.
static unsigned translateShiftImm(unsigned imm) {
  // lsr #32 and asr #32 exist, but should be encoded as a 0.
  assert((imm & ~0x1f) == 0 && "Invalid shift encoding");

  if (imm == 0)
    return 32;
  return imm;
}

/// Prints the shift value with an immediate value.
static void printRegImmShift(raw_ostream &O, ARM_AM::ShiftOpc ShOpc,
                          unsigned ShImm, bool UseMarkup) {
  if (ShOpc == ARM_AM::no_shift || (ShOpc == ARM_AM::lsl && !ShImm))
    return;
  O << ", ";

  assert (!(ShOpc == ARM_AM::ror && !ShImm) && "Cannot have ror #0");
  O << getShiftOpcStr(ShOpc);

  if (ShOpc != ARM_AM::rrx) {
    O << " ";
    if (UseMarkup)
      O << "<imm:";
    O << "#" << translateShiftImm(ShImm);
    if (UseMarkup)
      O << ">";
  }
}

ARMInstPrinter::ARMInstPrinter(const MCAsmInfo &MAI,
                               const MCInstrInfo &MII,
                               const MCRegisterInfo &MRI,
                               const MCSubtargetInfo &STI) :
  MCInstPrinter(MAI, MII, MRI) {
  // Initialize the set of available features.
  setAvailableFeatures(STI.getFeatureBits());
}

void ARMInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  OS << markup("<reg:")
     << getRegisterName(RegNo)
     << markup(">");
}

void ARMInstPrinter::printInst(const MCInst *MI, raw_ostream &O,
                               StringRef Annot) {
  unsigned Opcode = MI->getOpcode();

  // Check for HINT instructions w/ canonical names.
  if (Opcode == ARM::HINT || Opcode == ARM::t2HINT) {
    switch (MI->getOperand(0).getImm()) {
    case 0: O << "\tnop"; break;
    case 1: O << "\tyield"; break;
    case 2: O << "\twfe"; break;
    case 3: O << "\twfi"; break;
    case 4: O << "\tsev"; break;
    default:
      // Anything else should just print normally.
      printInstruction(MI, O);
      printAnnotation(O, Annot);
      return;
    }
    printPredicateOperand(MI, 1, O);
    if (Opcode == ARM::t2HINT)
      O << ".w";
    printAnnotation(O, Annot);
    return;
  }

  // Check for MOVs and print canonical forms, instead.
  if (Opcode == ARM::MOVsr) {
    // FIXME: Thumb variants?
    const MCOperand &Dst = MI->getOperand(0);
    const MCOperand &MO1 = MI->getOperand(1);
    const MCOperand &MO2 = MI->getOperand(2);
    const MCOperand &MO3 = MI->getOperand(3);

    O << '\t' << ARM_AM::getShiftOpcStr(ARM_AM::getSORegShOp(MO3.getImm()));
    printSBitModifierOperand(MI, 6, O);
    printPredicateOperand(MI, 4, O);

    O << '\t';
    printRegName(O, Dst.getReg());
    O << ", ";
    printRegName(O, MO1.getReg());

    O << ", ";
    printRegName(O, MO2.getReg());
    assert(ARM_AM::getSORegOffset(MO3.getImm()) == 0);
    printAnnotation(O, Annot);
    return;
  }

  if (Opcode == ARM::MOVsi) {
    // FIXME: Thumb variants?
    const MCOperand &Dst = MI->getOperand(0);
    const MCOperand &MO1 = MI->getOperand(1);
    const MCOperand &MO2 = MI->getOperand(2);

    O << '\t' << ARM_AM::getShiftOpcStr(ARM_AM::getSORegShOp(MO2.getImm()));
    printSBitModifierOperand(MI, 5, O);
    printPredicateOperand(MI, 3, O);

    O << '\t';
    printRegName(O, Dst.getReg());
    O << ", ";
    printRegName(O, MO1.getReg());

    if (ARM_AM::getSORegShOp(MO2.getImm()) == ARM_AM::rrx) {
      printAnnotation(O, Annot);
      return;
    }

    O << ", "
      << markup("<imm:")
      << "#" << translateShiftImm(ARM_AM::getSORegOffset(MO2.getImm()))
      << markup(">");
    printAnnotation(O, Annot);
    return;
  }


  // A8.6.123 PUSH
  if ((Opcode == ARM::STMDB_UPD || Opcode == ARM::t2STMDB_UPD) &&
      MI->getOperand(0).getReg() == ARM::SP &&
      MI->getNumOperands() > 5) {
    // Should only print PUSH if there are at least two registers in the list.
    O << '\t' << "push";
    printPredicateOperand(MI, 2, O);
    if (Opcode == ARM::t2STMDB_UPD)
      O << ".w";
    O << '\t';
    printRegisterList(MI, 4, O);
    printAnnotation(O, Annot);
    return;
  }
  if (Opcode == ARM::STR_PRE_IMM && MI->getOperand(2).getReg() == ARM::SP &&
      MI->getOperand(3).getImm() == -4) {
    O << '\t' << "push";
    printPredicateOperand(MI, 4, O);
    O << "\t{";
    printRegName(O, MI->getOperand(1).getReg());
    O << "}";
    printAnnotation(O, Annot);
    return;
  }

  // A8.6.122 POP
  if ((Opcode == ARM::LDMIA_UPD || Opcode == ARM::t2LDMIA_UPD) &&
      MI->getOperand(0).getReg() == ARM::SP &&
      MI->getNumOperands() > 5) {
    // Should only print POP if there are at least two registers in the list.
    O << '\t' << "pop";
    printPredicateOperand(MI, 2, O);
    if (Opcode == ARM::t2LDMIA_UPD)
      O << ".w";
    O << '\t';
    printRegisterList(MI, 4, O);
    printAnnotation(O, Annot);
    return;
  }
  if (Opcode == ARM::LDR_POST_IMM && MI->getOperand(2).getReg() == ARM::SP &&
      MI->getOperand(4).getImm() == 4) {
    O << '\t' << "pop";
    printPredicateOperand(MI, 5, O);
    O << "\t{";
    printRegName(O, MI->getOperand(0).getReg());
    O << "}";
    printAnnotation(O, Annot);
    return;
  }


  // A8.6.355 VPUSH
  if ((Opcode == ARM::VSTMSDB_UPD || Opcode == ARM::VSTMDDB_UPD) &&
      MI->getOperand(0).getReg() == ARM::SP) {
    O << '\t' << "vpush";
    printPredicateOperand(MI, 2, O);
    O << '\t';
    printRegisterList(MI, 4, O);
    printAnnotation(O, Annot);
    return;
  }

  // A8.6.354 VPOP
  if ((Opcode == ARM::VLDMSIA_UPD || Opcode == ARM::VLDMDIA_UPD) &&
      MI->getOperand(0).getReg() == ARM::SP) {
    O << '\t' << "vpop";
    printPredicateOperand(MI, 2, O);
    O << '\t';
    printRegisterList(MI, 4, O);
    printAnnotation(O, Annot);
    return;
  }

  if (Opcode == ARM::tLDMIA) {
    bool Writeback = true;
    unsigned BaseReg = MI->getOperand(0).getReg();
    for (unsigned i = 3; i < MI->getNumOperands(); ++i) {
      if (MI->getOperand(i).getReg() == BaseReg)
        Writeback = false;
    }

    O << "\tldm";

    printPredicateOperand(MI, 1, O);
    O << '\t';
    printRegName(O, BaseReg);
    if (Writeback) O << "!";
    O << ", ";
    printRegisterList(MI, 3, O);
    printAnnotation(O, Annot);
    return;
  }

  // Thumb1 NOP
  if (Opcode == ARM::tMOVr && MI->getOperand(0).getReg() == ARM::R8 &&
      MI->getOperand(1).getReg() == ARM::R8) {
    O << "\tnop";
    printPredicateOperand(MI, 2, O);
    printAnnotation(O, Annot);
    return;
  }

  printInstruction(MI, O);
  printAnnotation(O, Annot);
}

void ARMInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    unsigned Reg = Op.getReg();
    printRegName(O, Reg);
  } else if (Op.isImm()) {
    O << markup("<imm:")
      << '#' << Op.getImm()
      << markup(">");
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    // If a symbolic branch target was added as a constant expression then print
    // that address in hex. And only print 32 unsigned bits for the address.
    const MCConstantExpr *BranchTarget = dyn_cast<MCConstantExpr>(Op.getExpr());
    int64_t Address;
    if (BranchTarget && BranchTarget->EvaluateAsAbsolute(Address)) {
      O << "0x";
      O.write_hex((uint32_t)Address);
    }
    else {
      // Otherwise, just print the expression.
      O << *Op.getExpr();
    }
  }
}

void ARMInstPrinter::printThumbLdrLabelOperand(const MCInst *MI, unsigned OpNum,
                                               raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  if (MO1.isExpr())
    O << *MO1.getExpr();
  else if (MO1.isImm()) {
    O << markup("<mem:") << "[pc, "
      << markup("<imm:") << "#" << MO1.getImm()
      << markup(">]>", "]");
  }
  else
    llvm_unreachable("Unknown LDR label operand?");
}

// so_reg is a 4-operand unit corresponding to register forms of the A5.1
// "Addressing Mode 1 - Data-processing operands" forms.  This includes:
//    REG 0   0           - e.g. R5
//    REG REG 0,SH_OPC    - e.g. R5, ROR R3
//    REG 0   IMM,SH_OPC  - e.g. R5, LSL #3
void ARMInstPrinter::printSORegRegOperand(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  const MCOperand &MO3 = MI->getOperand(OpNum+2);

  printRegName(O, MO1.getReg());

  // Print the shift opc.
  ARM_AM::ShiftOpc ShOpc = ARM_AM::getSORegShOp(MO3.getImm());
  O << ", " << ARM_AM::getShiftOpcStr(ShOpc);
  if (ShOpc == ARM_AM::rrx)
    return;

  O << ' ';
  printRegName(O, MO2.getReg());
  assert(ARM_AM::getSORegOffset(MO3.getImm()) == 0);
}

void ARMInstPrinter::printSORegImmOperand(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  printRegName(O, MO1.getReg());

  // Print the shift opc.
  printRegImmShift(O, ARM_AM::getSORegShOp(MO2.getImm()),
                   ARM_AM::getSORegOffset(MO2.getImm()), UseMarkup);
}


//===--------------------------------------------------------------------===//
// Addressing Mode #2
//===--------------------------------------------------------------------===//

void ARMInstPrinter::printAM2PreOrOffsetIndexOp(const MCInst *MI, unsigned Op,
                                                raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op+1);
  const MCOperand &MO3 = MI->getOperand(Op+2);

  O << markup("<mem:") << "[";
  printRegName(O, MO1.getReg());

  if (!MO2.getReg()) {
    if (ARM_AM::getAM2Offset(MO3.getImm())) { // Don't print +0.
      O << ", "
        << markup("<imm:")
        << "#"
        << ARM_AM::getAddrOpcStr(ARM_AM::getAM2Op(MO3.getImm()))
        << ARM_AM::getAM2Offset(MO3.getImm())
        << markup(">");
    }
    O << "]" << markup(">");
    return;
  }

  O << ", ";
  O << ARM_AM::getAddrOpcStr(ARM_AM::getAM2Op(MO3.getImm()));
  printRegName(O, MO2.getReg());

  printRegImmShift(O, ARM_AM::getAM2ShiftOpc(MO3.getImm()),
                   ARM_AM::getAM2Offset(MO3.getImm()), UseMarkup);
  O << "]" << markup(">");
}

void ARMInstPrinter::printAddrModeTBB(const MCInst *MI, unsigned Op,
                                           raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op+1);
  O << markup("<mem:") << "[";
  printRegName(O, MO1.getReg());
  O << ", ";
  printRegName(O, MO2.getReg());
  O << "]" << markup(">");
}

void ARMInstPrinter::printAddrModeTBH(const MCInst *MI, unsigned Op,
                                           raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op+1);
  O << markup("<mem:") << "[";
  printRegName(O, MO1.getReg());
  O << ", ";
  printRegName(O, MO2.getReg());
  O << ", lsl " << markup("<imm:") << "#1" << markup(">") << "]" << markup(">");
}

void ARMInstPrinter::printAddrMode2Operand(const MCInst *MI, unsigned Op,
                                           raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op, O);
    return;
  }

#ifndef NDEBUG
  const MCOperand &MO3 = MI->getOperand(Op+2);
  unsigned IdxMode = ARM_AM::getAM2IdxMode(MO3.getImm());
  assert(IdxMode != ARMII::IndexModePost &&
         "Should be pre or offset index op");
#endif

  printAM2PreOrOffsetIndexOp(MI, Op, O);
}

void ARMInstPrinter::printAddrMode2OffsetOperand(const MCInst *MI,
                                                 unsigned OpNum,
                                                 raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  if (!MO1.getReg()) {
    unsigned ImmOffs = ARM_AM::getAM2Offset(MO2.getImm());
    O << markup("<imm:")
      << '#' << ARM_AM::getAddrOpcStr(ARM_AM::getAM2Op(MO2.getImm()))
      << ImmOffs
      << markup(">");
    return;
  }

  O << ARM_AM::getAddrOpcStr(ARM_AM::getAM2Op(MO2.getImm()));
  printRegName(O, MO1.getReg());

  printRegImmShift(O, ARM_AM::getAM2ShiftOpc(MO2.getImm()),
                   ARM_AM::getAM2Offset(MO2.getImm()), UseMarkup);
}

//===--------------------------------------------------------------------===//
// Addressing Mode #3
//===--------------------------------------------------------------------===//

void ARMInstPrinter::printAM3PostIndexOp(const MCInst *MI, unsigned Op,
                                         raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op+1);
  const MCOperand &MO3 = MI->getOperand(Op+2);

  O << markup("<mem:") << "[";
  printRegName(O, MO1.getReg());
  O << "], " << markup(">");

  if (MO2.getReg()) {
    O << (char)ARM_AM::getAM3Op(MO3.getImm());
    printRegName(O, MO2.getReg());
    return;
  }

  unsigned ImmOffs = ARM_AM::getAM3Offset(MO3.getImm());
  O << markup("<imm:")
    << '#'
    << ARM_AM::getAddrOpcStr(ARM_AM::getAM3Op(MO3.getImm()))
    << ImmOffs
    << markup(">");
}

void ARMInstPrinter::printAM3PreOrOffsetIndexOp(const MCInst *MI, unsigned Op,
                                                raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op+1);
  const MCOperand &MO3 = MI->getOperand(Op+2);

  O << markup("<mem:") << '[';
  printRegName(O, MO1.getReg());

  if (MO2.getReg()) {
    O << ", " << getAddrOpcStr(ARM_AM::getAM3Op(MO3.getImm()));
    printRegName(O, MO2.getReg());
    O << ']' << markup(">");
    return;
  }

  //If the op is sub we have to print the immediate even if it is 0
  unsigned ImmOffs = ARM_AM::getAM3Offset(MO3.getImm());
  ARM_AM::AddrOpc op = ARM_AM::getAM3Op(MO3.getImm());

  if (ImmOffs || (op == ARM_AM::sub)) {
    O << ", "
      << markup("<imm:")
      << "#"
      << ARM_AM::getAddrOpcStr(op)
      << ImmOffs
      << markup(">");
  }
  O << ']' << markup(">");
}

void ARMInstPrinter::printAddrMode3Operand(const MCInst *MI, unsigned Op,
                                           raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);
  if (!MO1.isReg()) {   //  For label symbolic references.
    printOperand(MI, Op, O);
    return;
  }

  const MCOperand &MO3 = MI->getOperand(Op+2);
  unsigned IdxMode = ARM_AM::getAM3IdxMode(MO3.getImm());

  if (IdxMode == ARMII::IndexModePost) {
    printAM3PostIndexOp(MI, Op, O);
    return;
  }
  printAM3PreOrOffsetIndexOp(MI, Op, O);
}

void ARMInstPrinter::printAddrMode3OffsetOperand(const MCInst *MI,
                                                 unsigned OpNum,
                                                 raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  if (MO1.getReg()) {
    O << getAddrOpcStr(ARM_AM::getAM3Op(MO2.getImm()));
    printRegName(O, MO1.getReg());
    return;
  }

  unsigned ImmOffs = ARM_AM::getAM3Offset(MO2.getImm());
  O << markup("<imm:")
    << '#' << ARM_AM::getAddrOpcStr(ARM_AM::getAM3Op(MO2.getImm())) << ImmOffs
    << markup(">");
}

void ARMInstPrinter::printPostIdxImm8Operand(const MCInst *MI,
                                             unsigned OpNum,
                                             raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  unsigned Imm = MO.getImm();
  O << markup("<imm:")
    << '#' << ((Imm & 256) ? "" : "-") << (Imm & 0xff)
    << markup(">");
}

void ARMInstPrinter::printPostIdxRegOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  O << (MO2.getImm() ? "" : "-");
  printRegName(O, MO1.getReg());
}

void ARMInstPrinter::printPostIdxImm8s4Operand(const MCInst *MI,
                                             unsigned OpNum,
                                             raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  unsigned Imm = MO.getImm();
  O << markup("<imm:")
    << '#' << ((Imm & 256) ? "" : "-") << ((Imm & 0xff) << 2)
    << markup(">");
}


void ARMInstPrinter::printLdStmModeOperand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  ARM_AM::AMSubMode Mode = ARM_AM::getAM4SubMode(MI->getOperand(OpNum)
                                                 .getImm());
  O << ARM_AM::getAMSubModeStr(Mode);
}

void ARMInstPrinter::printAddrMode5Operand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, OpNum, O);
    return;
  }

  O << markup("<mem:") << "[";
  printRegName(O, MO1.getReg());

  unsigned ImmOffs = ARM_AM::getAM5Offset(MO2.getImm());
  unsigned Op = ARM_AM::getAM5Op(MO2.getImm());
  if (ImmOffs || Op == ARM_AM::sub) {
    O << ", "
      << markup("<imm:")
      << "#"
      << ARM_AM::getAddrOpcStr(ARM_AM::getAM5Op(MO2.getImm()))
      << ImmOffs * 4
      << markup(">");
  }
  O << "]" << markup(">");
}

void ARMInstPrinter::printAddrMode6Operand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  O << markup("<mem:") << "[";
  printRegName(O, MO1.getReg());
  if (MO2.getImm()) {
    // FIXME: Both darwin as and GNU as violate ARM docs here.
    O << ", :" << (MO2.getImm() << 3);
  }
  O << "]" << markup(">");
}

void ARMInstPrinter::printAddrMode7Operand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  O << markup("<mem:") << "[";
  printRegName(O, MO1.getReg());
  O << "]" << markup(">");
}

void ARMInstPrinter::printAddrMode6OffsetOperand(const MCInst *MI,
                                                 unsigned OpNum,
                                                 raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  if (MO.getReg() == 0)
    O << "!";
  else {
    O << ", ";
    printRegName(O, MO.getReg());
  }
}

void ARMInstPrinter::printBitfieldInvMaskImmOperand(const MCInst *MI,
                                                    unsigned OpNum,
                                                    raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  uint32_t v = ~MO.getImm();
  int32_t lsb = CountTrailingZeros_32(v);
  int32_t width = (32 - CountLeadingZeros_32 (v)) - lsb;
  assert(MO.isImm() && "Not a valid bf_inv_mask_imm value!");
  O << markup("<imm:") << '#' << lsb << markup(">")
    << ", "
    << markup("<imm:") << '#' << width << markup(">");
}

void ARMInstPrinter::printMemBOption(const MCInst *MI, unsigned OpNum,
                                     raw_ostream &O) {
  unsigned val = MI->getOperand(OpNum).getImm();
  O << ARM_MB::MemBOptToString(val);
}

void ARMInstPrinter::printShiftImmOperand(const MCInst *MI, unsigned OpNum,
                                          raw_ostream &O) {
  unsigned ShiftOp = MI->getOperand(OpNum).getImm();
  bool isASR = (ShiftOp & (1 << 5)) != 0;
  unsigned Amt = ShiftOp & 0x1f;
  if (isASR) {
    O << ", asr "
      << markup("<imm:")
      << "#" << (Amt == 0 ? 32 : Amt)
      << markup(">");
  }
  else if (Amt) {
    O << ", lsl "
      << markup("<imm:")
      << "#" << Amt
      << markup(">");
  }
}

void ARMInstPrinter::printPKHLSLShiftImm(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNum).getImm();
  if (Imm == 0)
    return;
  assert(Imm > 0 && Imm < 32 && "Invalid PKH shift immediate value!");
  O << ", lsl " << markup("<imm:") << "#" << Imm << markup(">");
}

void ARMInstPrinter::printPKHASRShiftImm(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNum).getImm();
  // A shift amount of 32 is encoded as 0.
  if (Imm == 0)
    Imm = 32;
  assert(Imm > 0 && Imm <= 32 && "Invalid PKH shift immediate value!");
  O << ", asr " << markup("<imm:") << "#" << Imm << markup(">");
}

void ARMInstPrinter::printRegisterList(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  O << "{";
  for (unsigned i = OpNum, e = MI->getNumOperands(); i != e; ++i) {
    if (i != OpNum) O << ", ";
    printRegName(O, MI->getOperand(i).getReg());
  }
  O << "}";
}

void ARMInstPrinter::printSetendOperand(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);
  if (Op.getImm())
    O << "be";
  else
    O << "le";
}

void ARMInstPrinter::printCPSIMod(const MCInst *MI, unsigned OpNum,
                                  raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);
  O << ARM_PROC::IModToString(Op.getImm());
}

void ARMInstPrinter::printCPSIFlag(const MCInst *MI, unsigned OpNum,
                                   raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);
  unsigned IFlags = Op.getImm();
  for (int i=2; i >= 0; --i)
    if (IFlags & (1 << i))
      O << ARM_PROC::IFlagsToString(1 << i);

  if (IFlags == 0)
    O << "none";
}

void ARMInstPrinter::printMSRMaskOperand(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNum);
  unsigned SpecRegRBit = Op.getImm() >> 4;
  unsigned Mask = Op.getImm() & 0xf;

  if (getAvailableFeatures() & ARM::FeatureMClass) {
    unsigned SYSm = Op.getImm();
    unsigned Opcode = MI->getOpcode();
    // For reads of the special registers ignore the "mask encoding" bits
    // which are only for writes.
    if (Opcode == ARM::t2MRS_M)
      SYSm &= 0xff;
    switch (SYSm) {
    default: llvm_unreachable("Unexpected mask value!");
    case     0:
    case 0x800: O << "apsr"; return; // with _nzcvq bits is an alias for aspr
    case 0x400: O << "apsr_g"; return;
    case 0xc00: O << "apsr_nzcvqg"; return;
    case     1:
    case 0x801: O << "iapsr"; return; // with _nzcvq bits is an alias for iapsr
    case 0x401: O << "iapsr_g"; return;
    case 0xc01: O << "iapsr_nzcvqg"; return;
    case     2:
    case 0x802: O << "eapsr"; return; // with _nzcvq bits is an alias for eapsr
    case 0x402: O << "eapsr_g"; return;
    case 0xc02: O << "eapsr_nzcvqg"; return;
    case     3:
    case 0x803: O << "xpsr"; return; // with _nzcvq bits is an alias for xpsr
    case 0x403: O << "xpsr_g"; return;
    case 0xc03: O << "xpsr_nzcvqg"; return;
    case     5:
    case 0x805: O << "ipsr"; return;
    case     6:
    case 0x806: O << "epsr"; return;
    case     7:
    case 0x807: O << "iepsr"; return;
    case     8:
    case 0x808: O << "msp"; return;
    case     9:
    case 0x809: O << "psp"; return;
    case  0x10:
    case 0x810: O << "primask"; return;
    case  0x11:
    case 0x811: O << "basepri"; return;
    case  0x12:
    case 0x812: O << "basepri_max"; return;
    case  0x13:
    case 0x813: O << "faultmask"; return;
    case  0x14:
    case 0x814: O << "control"; return;
    }
  }

  // As special cases, CPSR_f, CPSR_s and CPSR_fs prefer printing as
  // APSR_nzcvq, APSR_g and APSRnzcvqg, respectively.
  if (!SpecRegRBit && (Mask == 8 || Mask == 4 || Mask == 12)) {
    O << "APSR_";
    switch (Mask) {
    default: llvm_unreachable("Unexpected mask value!");
    case 4:  O << "g"; return;
    case 8:  O << "nzcvq"; return;
    case 12: O << "nzcvqg"; return;
    }
  }

  if (SpecRegRBit)
    O << "SPSR";
  else
    O << "CPSR";

  if (Mask) {
    O << '_';
    if (Mask & 8) O << 'f';
    if (Mask & 4) O << 's';
    if (Mask & 2) O << 'x';
    if (Mask & 1) O << 'c';
  }
}

void ARMInstPrinter::printPredicateOperand(const MCInst *MI, unsigned OpNum,
                                           raw_ostream &O) {
  ARMCC::CondCodes CC = (ARMCC::CondCodes)MI->getOperand(OpNum).getImm();
  // Handle the undefined 15 CC value here for printing so we don't abort().
  if ((unsigned)CC == 15)
    O << "<und>";
  else if (CC != ARMCC::AL)
    O << ARMCondCodeToString(CC);
}

void ARMInstPrinter::printMandatoryPredicateOperand(const MCInst *MI,
                                                    unsigned OpNum,
                                                    raw_ostream &O) {
  ARMCC::CondCodes CC = (ARMCC::CondCodes)MI->getOperand(OpNum).getImm();
  O << ARMCondCodeToString(CC);
}

void ARMInstPrinter::printSBitModifierOperand(const MCInst *MI, unsigned OpNum,
                                              raw_ostream &O) {
  if (MI->getOperand(OpNum).getReg()) {
    assert(MI->getOperand(OpNum).getReg() == ARM::CPSR &&
           "Expect ARM CPSR register!");
    O << 's';
  }
}

void ARMInstPrinter::printNoHashImmediate(const MCInst *MI, unsigned OpNum,
                                          raw_ostream &O) {
  O << MI->getOperand(OpNum).getImm();
}

void ARMInstPrinter::printPImmediate(const MCInst *MI, unsigned OpNum,
                                     raw_ostream &O) {
  O << "p" << MI->getOperand(OpNum).getImm();
}

void ARMInstPrinter::printCImmediate(const MCInst *MI, unsigned OpNum,
                                     raw_ostream &O) {
  O << "c" << MI->getOperand(OpNum).getImm();
}

void ARMInstPrinter::printCoprocOptionImm(const MCInst *MI, unsigned OpNum,
                                          raw_ostream &O) {
  O << "{" << MI->getOperand(OpNum).getImm() << "}";
}

void ARMInstPrinter::printPCLabel(const MCInst *MI, unsigned OpNum,
                                  raw_ostream &O) {
  llvm_unreachable("Unhandled PC-relative pseudo-instruction!");
}

void ARMInstPrinter::printAdrLabelOperand(const MCInst *MI, unsigned OpNum,
                                  raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);

  if (MO.isExpr()) {
    O << *MO.getExpr();
    return;
  }

  int32_t OffImm = (int32_t)MO.getImm();

  O << markup("<imm:");
  if (OffImm == INT32_MIN)
    O << "#-0";
  else if (OffImm < 0)
    O << "#-" << -OffImm;
  else
    O << "#" << OffImm;
  O << markup(">");
}

void ARMInstPrinter::printThumbS4ImmOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  O << markup("<imm:")
    << "#" << MI->getOperand(OpNum).getImm() * 4
    << markup(">");
}

void ARMInstPrinter::printThumbSRImm(const MCInst *MI, unsigned OpNum,
                                     raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNum).getImm();
  O << markup("<imm:")
    << "#" << (Imm == 0 ? 32 : Imm)
    << markup(">");
}

void ARMInstPrinter::printThumbITMask(const MCInst *MI, unsigned OpNum,
                                      raw_ostream &O) {
  // (3 - the number of trailing zeros) is the number of then / else.
  unsigned Mask = MI->getOperand(OpNum).getImm();
  unsigned Firstcond = MI->getOperand(OpNum-1).getImm();
  unsigned CondBit0 = Firstcond & 1;
  unsigned NumTZ = CountTrailingZeros_32(Mask);
  assert(NumTZ <= 3 && "Invalid IT mask!");
  for (unsigned Pos = 3, e = NumTZ; Pos > e; --Pos) {
    bool T = ((Mask >> Pos) & 1) == CondBit0;
    if (T)
      O << 't';
    else
      O << 'e';
  }
}

void ARMInstPrinter::printThumbAddrModeRROperand(const MCInst *MI, unsigned Op,
                                                 raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op + 1);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op, O);
    return;
  }

  O << markup("<mem:") << "[";
  printRegName(O, MO1.getReg());
  if (unsigned RegNum = MO2.getReg()) {
    O << ", ";
    printRegName(O, RegNum);
  }
  O << "]" << markup(">");
}

void ARMInstPrinter::printThumbAddrModeImm5SOperand(const MCInst *MI,
                                                    unsigned Op,
                                                    raw_ostream &O,
                                                    unsigned Scale) {
  const MCOperand &MO1 = MI->getOperand(Op);
  const MCOperand &MO2 = MI->getOperand(Op + 1);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, Op, O);
    return;
  }

  O << markup("<mem:") << "[";
  printRegName(O, MO1.getReg());
  if (unsigned ImmOffs = MO2.getImm()) {
    O << ", "
      << markup("<imm:")
      << "#" << ImmOffs * Scale
      << markup(">");
  }
  O << "]" << markup(">");
}

void ARMInstPrinter::printThumbAddrModeImm5S1Operand(const MCInst *MI,
                                                     unsigned Op,
                                                     raw_ostream &O) {
  printThumbAddrModeImm5SOperand(MI, Op, O, 1);
}

void ARMInstPrinter::printThumbAddrModeImm5S2Operand(const MCInst *MI,
                                                     unsigned Op,
                                                     raw_ostream &O) {
  printThumbAddrModeImm5SOperand(MI, Op, O, 2);
}

void ARMInstPrinter::printThumbAddrModeImm5S4Operand(const MCInst *MI,
                                                     unsigned Op,
                                                     raw_ostream &O) {
  printThumbAddrModeImm5SOperand(MI, Op, O, 4);
}

void ARMInstPrinter::printThumbAddrModeSPOperand(const MCInst *MI, unsigned Op,
                                                 raw_ostream &O) {
  printThumbAddrModeImm5SOperand(MI, Op, O, 4);
}

// Constant shifts t2_so_reg is a 2-operand unit corresponding to the Thumb2
// register with shift forms.
// REG 0   0           - e.g. R5
// REG IMM, SH_OPC     - e.g. R5, LSL #3
void ARMInstPrinter::printT2SOOperand(const MCInst *MI, unsigned OpNum,
                                      raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  unsigned Reg = MO1.getReg();
  printRegName(O, Reg);

  // Print the shift opc.
  assert(MO2.isImm() && "Not a valid t2_so_reg value!");
  printRegImmShift(O, ARM_AM::getSORegShOp(MO2.getImm()),
                   ARM_AM::getSORegOffset(MO2.getImm()), UseMarkup);
}

void ARMInstPrinter::printAddrModeImm12Operand(const MCInst *MI, unsigned OpNum,
                                               raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  if (!MO1.isReg()) {   // FIXME: This is for CP entries, but isn't right.
    printOperand(MI, OpNum, O);
    return;
  }

  O << markup("<mem:") << "[";
  printRegName(O, MO1.getReg());

  int32_t OffImm = (int32_t)MO2.getImm();
  bool isSub = OffImm < 0;
  // Special value for #-0. All others are normal.
  if (OffImm == INT32_MIN)
    OffImm = 0;
  if (isSub) {
    O << ", "
      << markup("<imm:") 
      << "#-" << -OffImm
      << markup(">");
  }
  else if (OffImm > 0) {
    O << ", "
      << markup("<imm:") 
      << "#" << OffImm
      << markup(">");
  }
  O << "]" << markup(">");
}

void ARMInstPrinter::printT2AddrModeImm8Operand(const MCInst *MI,
                                                unsigned OpNum,
                                                raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  O << markup("<mem:") << "[";
  printRegName(O, MO1.getReg());

  int32_t OffImm = (int32_t)MO2.getImm();
  // Don't print +0.
  if (OffImm != 0)
    O << ", ";
  if (OffImm != 0 && UseMarkup)
    O << "<imm:";
  if (OffImm == INT32_MIN)
    O << "#-0";
  else if (OffImm < 0)
    O << "#-" << -OffImm;
  else if (OffImm > 0)
    O << "#" << OffImm;
  if (OffImm != 0 && UseMarkup)
    O << ">";
  O << "]" << markup(">");
}

void ARMInstPrinter::printT2AddrModeImm8s4Operand(const MCInst *MI,
                                                  unsigned OpNum,
                                                  raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  if (!MO1.isReg()) {   //  For label symbolic references.
    printOperand(MI, OpNum, O);
    return;
  }

  O << markup("<mem:") << "[";
  printRegName(O, MO1.getReg());

  int32_t OffImm = (int32_t)MO2.getImm();

  assert(((OffImm & 0x3) == 0) && "Not a valid immediate!");

  // Don't print +0.
  if (OffImm != 0)
    O << ", ";
  if (OffImm != 0 && UseMarkup)
    O << "<imm:";
  if (OffImm == INT32_MIN)
    O << "#-0";
  else if (OffImm < 0)
    O << "#-" << -OffImm;
  else if (OffImm > 0)
    O << "#" << OffImm;
  if (OffImm != 0 && UseMarkup)
    O << ">";
  O << "]" << markup(">");
}

void ARMInstPrinter::printT2AddrModeImm0_1020s4Operand(const MCInst *MI,
                                                       unsigned OpNum,
                                                       raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);

  O << markup("<mem:") << "[";
  printRegName(O, MO1.getReg());
  if (MO2.getImm()) {
    O << ", "
      << markup("<imm:")
      << "#" << MO2.getImm() * 4
      << markup(">");
  }
  O << "]" << markup(">");
}

void ARMInstPrinter::printT2AddrModeImm8OffsetOperand(const MCInst *MI,
                                                      unsigned OpNum,
                                                      raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  int32_t OffImm = (int32_t)MO1.getImm();
  O << ", " << markup("<imm:");
  if (OffImm < 0)
    O << "#-" << -OffImm;
  else
    O << "#" << OffImm;
  O << markup(">");
}

void ARMInstPrinter::printT2AddrModeImm8s4OffsetOperand(const MCInst *MI,
                                                        unsigned OpNum,
                                                        raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  int32_t OffImm = (int32_t)MO1.getImm();

  assert(((OffImm & 0x3) == 0) && "Not a valid immediate!");

  // Don't print +0.
  if (OffImm != 0)
    O << ", ";
  if (OffImm != 0 && UseMarkup)
    O << "<imm:";
  if (OffImm == INT32_MIN)
    O << "#-0";
  else if (OffImm < 0)
    O << "#-" << -OffImm;
  else if (OffImm > 0)
    O << "#" << OffImm;
  if (OffImm != 0 && UseMarkup)
    O << ">";
}

void ARMInstPrinter::printT2AddrModeSoRegOperand(const MCInst *MI,
                                                 unsigned OpNum,
                                                 raw_ostream &O) {
  const MCOperand &MO1 = MI->getOperand(OpNum);
  const MCOperand &MO2 = MI->getOperand(OpNum+1);
  const MCOperand &MO3 = MI->getOperand(OpNum+2);

  O << markup("<mem:") << "[";
  printRegName(O, MO1.getReg());

  assert(MO2.getReg() && "Invalid so_reg load / store address!");
  O << ", ";
  printRegName(O, MO2.getReg());

  unsigned ShAmt = MO3.getImm();
  if (ShAmt) {
    assert(ShAmt <= 3 && "Not a valid Thumb2 addressing mode!");
    O << ", lsl "
      << markup("<imm:")
      << "#" << ShAmt
      << markup(">");
  }
  O << "]" << markup(">");
}

void ARMInstPrinter::printFPImmOperand(const MCInst *MI, unsigned OpNum,
                                       raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  O << markup("<imm:")
    << '#' << ARM_AM::getFPImmFloat(MO.getImm())
    << markup(">");
}

void ARMInstPrinter::printNEONModImmOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  unsigned EncodedImm = MI->getOperand(OpNum).getImm();
  unsigned EltBits;
  uint64_t Val = ARM_AM::decodeNEONModImm(EncodedImm, EltBits);
  O << markup("<imm:")
    << "#0x";
  O.write_hex(Val);
  O << markup(">");
}

void ARMInstPrinter::printImmPlusOneOperand(const MCInst *MI, unsigned OpNum,
                                            raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNum).getImm();
  O << markup("<imm:")
    << "#" << Imm + 1
    << markup(">");
}

void ARMInstPrinter::printRotImmOperand(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNum).getImm();
  if (Imm == 0)
    return;
  O << ", ror "
    << markup("<imm:")
    << "#";
  switch (Imm) {
  default: assert (0 && "illegal ror immediate!");
  case 1: O << "8"; break;
  case 2: O << "16"; break;
  case 3: O << "24"; break;
  }
  O << markup(">");
}

void ARMInstPrinter::printFBits16(const MCInst *MI, unsigned OpNum,
                                  raw_ostream &O) {
  O << markup("<imm:")
    << "#" << 16 - MI->getOperand(OpNum).getImm()
    << markup(">");
}

void ARMInstPrinter::printFBits32(const MCInst *MI, unsigned OpNum,
                                  raw_ostream &O) {
  O << markup("<imm:")
    << "#" << 32 - MI->getOperand(OpNum).getImm()
    << markup(">");
}

void ARMInstPrinter::printVectorIndex(const MCInst *MI, unsigned OpNum,
                                      raw_ostream &O) {
  O << "[" << MI->getOperand(OpNum).getImm() << "]";
}

void ARMInstPrinter::printVectorListOne(const MCInst *MI, unsigned OpNum,
                                        raw_ostream &O) {
  O << "{";
  printRegName(O, MI->getOperand(OpNum).getReg());
  O << "}";
}

void ARMInstPrinter::printVectorListTwo(const MCInst *MI, unsigned OpNum,
                                          raw_ostream &O) {
  unsigned Reg = MI->getOperand(OpNum).getReg();
  unsigned Reg0 = MRI.getSubReg(Reg, ARM::dsub_0);
  unsigned Reg1 = MRI.getSubReg(Reg, ARM::dsub_1);
  O << "{";
  printRegName(O, Reg0);
  O << ", ";
  printRegName(O, Reg1);
  O << "}";
}

void ARMInstPrinter::printVectorListTwoSpaced(const MCInst *MI,
                                              unsigned OpNum,
                                              raw_ostream &O) {
  unsigned Reg = MI->getOperand(OpNum).getReg();
  unsigned Reg0 = MRI.getSubReg(Reg, ARM::dsub_0);
  unsigned Reg1 = MRI.getSubReg(Reg, ARM::dsub_2);
  O << "{";
  printRegName(O, Reg0);
  O << ", ";
  printRegName(O, Reg1);
  O << "}";
}

void ARMInstPrinter::printVectorListThree(const MCInst *MI, unsigned OpNum,
                                          raw_ostream &O) {
  // Normally, it's not safe to use register enum values directly with
  // addition to get the next register, but for VFP registers, the
  // sort order is guaranteed because they're all of the form D<n>.
  O << "{";
  printRegName(O, MI->getOperand(OpNum).getReg());
  O << ", ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 1);
  O << ", ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 2);
  O << "}";
}

void ARMInstPrinter::printVectorListFour(const MCInst *MI, unsigned OpNum,
                                         raw_ostream &O) {
  // Normally, it's not safe to use register enum values directly with
  // addition to get the next register, but for VFP registers, the
  // sort order is guaranteed because they're all of the form D<n>.
  O << "{";
  printRegName(O, MI->getOperand(OpNum).getReg());
  O << ", ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 1);
  O << ", ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 2);
  O << ", ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 3);
  O << "}";
}

void ARMInstPrinter::printVectorListOneAllLanes(const MCInst *MI,
                                                unsigned OpNum,
                                                raw_ostream &O) {
  O << "{";
  printRegName(O, MI->getOperand(OpNum).getReg());
  O << "[]}";
}

void ARMInstPrinter::printVectorListTwoAllLanes(const MCInst *MI,
                                                unsigned OpNum,
                                                raw_ostream &O) {
  unsigned Reg = MI->getOperand(OpNum).getReg();
  unsigned Reg0 = MRI.getSubReg(Reg, ARM::dsub_0);
  unsigned Reg1 = MRI.getSubReg(Reg, ARM::dsub_1);
  O << "{";
  printRegName(O, Reg0);
  O << "[], ";
  printRegName(O, Reg1);
  O << "[]}";
}

void ARMInstPrinter::printVectorListThreeAllLanes(const MCInst *MI,
                                                  unsigned OpNum,
                                                  raw_ostream &O) {
  // Normally, it's not safe to use register enum values directly with
  // addition to get the next register, but for VFP registers, the
  // sort order is guaranteed because they're all of the form D<n>.
  O << "{";
  printRegName(O, MI->getOperand(OpNum).getReg());
  O << "[], ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 1);
  O << "[], ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 2);
  O << "[]}";
}

void ARMInstPrinter::printVectorListFourAllLanes(const MCInst *MI,
                                                  unsigned OpNum,
                                                  raw_ostream &O) {
  // Normally, it's not safe to use register enum values directly with
  // addition to get the next register, but for VFP registers, the
  // sort order is guaranteed because they're all of the form D<n>.
  O << "{";
  printRegName(O, MI->getOperand(OpNum).getReg());
  O << "[], ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 1);
  O << "[], ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 2);
  O << "[], ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 3);
  O << "[]}";
}

void ARMInstPrinter::printVectorListTwoSpacedAllLanes(const MCInst *MI,
                                                      unsigned OpNum,
                                                      raw_ostream &O) {
  unsigned Reg = MI->getOperand(OpNum).getReg();
  unsigned Reg0 = MRI.getSubReg(Reg, ARM::dsub_0);
  unsigned Reg1 = MRI.getSubReg(Reg, ARM::dsub_2);
  O << "{";
  printRegName(O, Reg0);
  O << "[], ";
  printRegName(O, Reg1);
  O << "[]}";
}

void ARMInstPrinter::printVectorListThreeSpacedAllLanes(const MCInst *MI,
                                                        unsigned OpNum,
                                                        raw_ostream &O) {
  // Normally, it's not safe to use register enum values directly with
  // addition to get the next register, but for VFP registers, the
  // sort order is guaranteed because they're all of the form D<n>.
  O << "{";
  printRegName(O, MI->getOperand(OpNum).getReg());
  O  << "[], ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 2);
  O << "[], ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 4);
  O << "[]}";
}

void ARMInstPrinter::printVectorListFourSpacedAllLanes(const MCInst *MI,
                                                       unsigned OpNum,
                                                       raw_ostream &O) {
  // Normally, it's not safe to use register enum values directly with
  // addition to get the next register, but for VFP registers, the
  // sort order is guaranteed because they're all of the form D<n>.
  O << "{";
  printRegName(O, MI->getOperand(OpNum).getReg());
  O << "[], ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 2);
  O << "[], ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 4);
  O << "[], ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 6);
  O << "[]}";
}

void ARMInstPrinter::printVectorListThreeSpaced(const MCInst *MI,
                                                unsigned OpNum,
                                                raw_ostream &O) {
  // Normally, it's not safe to use register enum values directly with
  // addition to get the next register, but for VFP registers, the
  // sort order is guaranteed because they're all of the form D<n>.
  O << "{";
  printRegName(O, MI->getOperand(OpNum).getReg());
  O << ", ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 2);
  O << ", ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 4);
  O << "}";
}

void ARMInstPrinter::printVectorListFourSpaced(const MCInst *MI,
                                                unsigned OpNum,
                                                raw_ostream &O) {
  // Normally, it's not safe to use register enum values directly with
  // addition to get the next register, but for VFP registers, the
  // sort order is guaranteed because they're all of the form D<n>.
  O << "{";
  printRegName(O, MI->getOperand(OpNum).getReg());
  O << ", ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 2);
  O << ", ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 4);
  O << ", ";
  printRegName(O, MI->getOperand(OpNum).getReg() + 6);
  O << "}";
}
