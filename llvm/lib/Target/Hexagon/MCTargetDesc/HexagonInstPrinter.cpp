//===- HexagonInstPrinter.cpp - Convert Hexagon MCInst to assembly syntax -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class prints an Hexagon MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "HexagonAsmPrinter.h"
#include "HexagonInstPrinter.h"
#include "MCTargetDesc/HexagonBaseInfo.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#define GET_INSTRUCTION_NAME
#include "HexagonGenAsmWriter.inc"

HexagonInstPrinter::HexagonInstPrinter(MCAsmInfo const &MAI,
                                       MCInstrInfo const &MII,
                                       MCRegisterInfo const &MRI)
    : MCInstPrinter(MAI, MII, MRI), MII(MII), HasExtender(false) {
}

StringRef HexagonInstPrinter::getOpcodeName(unsigned Opcode) const {
  return MII.getName(Opcode);
}

void HexagonInstPrinter::printRegName(raw_ostream &O, unsigned RegNo) const {
  O << getRegName(RegNo);
}

StringRef HexagonInstPrinter::getRegName(unsigned RegNo) const {
  return getRegisterName(RegNo);
}

void HexagonInstPrinter::setExtender(MCInst const &MCI) {
  HasExtender = HexagonMCInstrInfo::isImmext(MCI);
}

void HexagonInstPrinter::printInst(const MCInst *MI, raw_ostream &OS,
                                   StringRef Annot, const MCSubtargetInfo &STI) {
  assert(HexagonMCInstrInfo::isBundle(*MI));
  assert(HexagonMCInstrInfo::bundleSize(*MI) <= HEXAGON_PACKET_SIZE);
  assert(HexagonMCInstrInfo::bundleSize(*MI) > 0);
  HasExtender = false;
  for (auto const &I : HexagonMCInstrInfo::bundleInstructions(*MI)) {
    MCInst const &MCI = *I.getInst();
    if (HexagonMCInstrInfo::isDuplex(MII, MCI)) {
      printInstruction(MCI.getOperand(1).getInst(), OS);
      OS << '\v';
      HasExtender = false;
      printInstruction(MCI.getOperand(0).getInst(), OS);
    } else
      printInstruction(&MCI, OS);
    setExtender(MCI);
    OS << "\n";
  }

  auto Separator = "";
  if (HexagonMCInstrInfo::isInnerLoop(*MI)) {
    OS << Separator;
    Separator = " ";
    MCInst ME;
    ME.setOpcode(Hexagon::ENDLOOP0);
    printInstruction(&ME, OS);
  }
  if (HexagonMCInstrInfo::isOuterLoop(*MI)) {
    OS << Separator;
    MCInst ME;
    ME.setOpcode(Hexagon::ENDLOOP1);
    printInstruction(&ME, OS);
  }
}

void HexagonInstPrinter::printOperand(MCInst const *MI, unsigned OpNo,
                                      raw_ostream &O) const {
  if (HexagonMCInstrInfo::getExtendableOp(MII, *MI) == OpNo &&
      (HasExtender || HexagonMCInstrInfo::isConstExtended(MII, *MI)))
    O << "#";
  MCOperand const &MO = MI->getOperand(OpNo);
  if (MO.isReg()) {
    O << getRegisterName(MO.getReg());
  } else if (MO.isExpr()) {
    int64_t Value;
    if (MO.getExpr()->evaluateAsAbsolute(Value))
      O << formatImm(Value);
    else
      O << *MO.getExpr();
  } else {
    llvm_unreachable("Unknown operand");
  }
}

void HexagonInstPrinter::printExtOperand(MCInst const *MI, unsigned OpNo,
                                         raw_ostream &O) const {
  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printUnsignedImmOperand(MCInst const *MI,
                                                 unsigned OpNo,
                                                 raw_ostream &O) const {
  O << MI->getOperand(OpNo).getImm();
}

void HexagonInstPrinter::printNegImmOperand(MCInst const *MI, unsigned OpNo,
                                            raw_ostream &O) const {
  O << -MI->getOperand(OpNo).getImm();
}

void HexagonInstPrinter::printNOneImmOperand(MCInst const *MI, unsigned OpNo,
                                             raw_ostream &O) const {
  O << -1;
}

void HexagonInstPrinter::prints3_6ImmOperand(MCInst const *MI, unsigned OpNo,
                                             raw_ostream &O) const {
  int64_t Imm;
  bool Success = MI->getOperand(OpNo).getExpr()->evaluateAsAbsolute(Imm);
  Imm = SignExtend64<9>(Imm);
  assert(Success); (void)Success;
  assert(((Imm & 0x3f) == 0) && "Lower 6 bits must be ZERO.");
  O << formatImm(Imm/64);
}

void HexagonInstPrinter::prints3_7ImmOperand(MCInst const *MI, unsigned OpNo,
                                             raw_ostream &O) const {
  int64_t Imm;
  bool Success = MI->getOperand(OpNo).getExpr()->evaluateAsAbsolute(Imm);
  Imm = SignExtend64<10>(Imm);
  assert(Success); (void)Success;
  assert(((Imm & 0x7f) == 0) && "Lower 7 bits must be ZERO.");
  O << formatImm(Imm/128);
}

void HexagonInstPrinter::prints4_6ImmOperand(MCInst const *MI, unsigned OpNo,
                                             raw_ostream &O) const {
  int64_t Imm;
  bool Success = MI->getOperand(OpNo).getExpr()->evaluateAsAbsolute(Imm);
  Imm = SignExtend64<10>(Imm);
  assert(Success); (void)Success;
  assert(((Imm & 0x3f) == 0) && "Lower 6 bits must be ZERO.");
  O << formatImm(Imm/64);
}

void HexagonInstPrinter::prints4_7ImmOperand(MCInst const *MI, unsigned OpNo,
                                             raw_ostream &O) const {
  int64_t Imm;
  bool Success = MI->getOperand(OpNo).getExpr()->evaluateAsAbsolute(Imm);
  Imm = SignExtend64<11>(Imm);
  assert(Success); (void)Success;
  assert(((Imm & 0x7f) == 0) && "Lower 7 bits must be ZERO.");
  O << formatImm(Imm/128);
}

void HexagonInstPrinter::printGlobalOperand(MCInst const *MI, unsigned OpNo,
                                            raw_ostream &O) const {
  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printJumpTable(MCInst const *MI, unsigned OpNo,
                                        raw_ostream &O) const {
  assert(MI->getOperand(OpNo).isExpr() && "Expecting expression");

  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printConstantPool(MCInst const *MI, unsigned OpNo,
                                           raw_ostream &O) const {
  assert(MI->getOperand(OpNo).isExpr() && "Expecting expression");

  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printBranchOperand(MCInst const *MI, unsigned OpNo,
                                            raw_ostream &O) const {
  // Branches can take an immediate operand.  This is used by the branch
  // selection pass to print $+8, an eight byte displacement from the PC.
  llvm_unreachable("Unknown branch operand.");
}

void HexagonInstPrinter::printCallOperand(MCInst const *MI, unsigned OpNo,
                                          raw_ostream &O) const {}

void HexagonInstPrinter::printAbsAddrOperand(MCInst const *MI, unsigned OpNo,
                                             raw_ostream &O) const {}

void HexagonInstPrinter::printPredicateOperand(MCInst const *MI, unsigned OpNo,
                                               raw_ostream &O) const {}

void HexagonInstPrinter::printSymbol(MCInst const *MI, unsigned OpNo,
                                     raw_ostream &O, bool hi) const {
  assert(MI->getOperand(OpNo).isImm() && "Unknown symbol operand");

  O << '#' << (hi ? "HI" : "LO") << '(';
  O << '#';
  printOperand(MI, OpNo, O);
  O << ')';
}

void HexagonInstPrinter::printBrtarget(MCInst const *MI, unsigned OpNo,
                                       raw_ostream &O) const {
  MCOperand const &MO = MI->getOperand(OpNo);
  assert (MO.isExpr());
  MCExpr const &Expr = *MO.getExpr();
  int64_t Value;
  if (Expr.evaluateAsAbsolute(Value))
    O << format("0x%" PRIx64, Value);
  else {
    if (HasExtender || HexagonMCInstrInfo::isConstExtended(MII, *MI))
      if (HexagonMCInstrInfo::getExtendableOp(MII, *MI) == OpNo)
        O << "##";
    O << Expr;
  }
}
