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
#include "Hexagon.h"
#include "HexagonInstPrinter.h"
#include "MCTargetDesc/HexagonMCInstrInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#define GET_INSTRUCTION_NAME
#include "HexagonGenAsmWriter.inc"

// Return the minimum value that a constant extendable operand can have
// without being extended.
static int getMinValue(uint64_t TSFlags) {
  unsigned isSigned =
      (TSFlags >> HexagonII::ExtentSignedPos) & HexagonII::ExtentSignedMask;
  unsigned bits =
      (TSFlags >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask;

  if (isSigned)
    return -1U << (bits - 1);

  return 0;
}

// Return the maximum value that a constant extendable operand can have
// without being extended.
static int getMaxValue(uint64_t TSFlags) {
  unsigned isSigned =
      (TSFlags >> HexagonII::ExtentSignedPos) & HexagonII::ExtentSignedMask;
  unsigned bits =
      (TSFlags >> HexagonII::ExtentBitsPos) & HexagonII::ExtentBitsMask;

  if (isSigned)
    return ~(-1U << (bits - 1));

  return ~(-1U << bits);
}

// Return true if the instruction must be extended.
static bool isExtended(uint64_t TSFlags) {
  return (TSFlags >> HexagonII::ExtendedPos) & HexagonII::ExtendedMask;
}

// Currently just used in an assert statement
static bool isExtendable(uint64_t TSFlags) LLVM_ATTRIBUTE_UNUSED;
// Return true if the instruction may be extended based on the operand value.
static bool isExtendable(uint64_t TSFlags) {
  return (TSFlags >> HexagonII::ExtendablePos) & HexagonII::ExtendableMask;
}

StringRef HexagonInstPrinter::getOpcodeName(unsigned Opcode) const {
  return MII.getName(Opcode);
}

void HexagonInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  OS << getRegisterName(RegNo);
}

void HexagonInstPrinter::setExtender(MCInst const &MCI) {
  HasExtender = HexagonMCInstrInfo::isImmext(MCI);
}

void HexagonInstPrinter::printInst(MCInst const *MI, raw_ostream &OS,
                                   StringRef Annot,
                                   MCSubtargetInfo const &STI) {
  assert(HexagonMCInstrInfo::isBundle(*MI));
  assert(HexagonMCInstrInfo::bundleSize(*MI) <= HEXAGON_PACKET_SIZE);
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
    Separator = " ";
    MCInst ME;
    ME.setOpcode(Hexagon::ENDLOOP1);
    printInstruction(&ME, OS);
  }
}

void HexagonInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O) const {
  const MCOperand& MO = MI->getOperand(OpNo);

  if (MO.isReg()) {
    printRegName(O, MO.getReg());
  } else if(MO.isExpr()) {
    MO.getExpr()->print(O, &MAI);
  } else if(MO.isImm()) {
    printImmOperand(MI, OpNo, O);
  } else {
    llvm_unreachable("Unknown operand");
  }
}

void HexagonInstPrinter::printImmOperand(const MCInst *MI, unsigned OpNo,
                                         raw_ostream &O) const {
  const MCOperand& MO = MI->getOperand(OpNo);

  if(MO.isExpr()) {
    MO.getExpr()->print(O, &MAI);
  } else if(MO.isImm()) {
    O << MI->getOperand(OpNo).getImm();
  } else {
    llvm_unreachable("Unknown operand");
  }
}

void HexagonInstPrinter::printExtOperand(const MCInst *MI, unsigned OpNo,
                                         raw_ostream &O) const {
  const MCOperand &MO = MI->getOperand(OpNo);
  const MCInstrDesc &MII = getMII().get(MI->getOpcode());

  assert((isExtendable(MII.TSFlags) || isExtended(MII.TSFlags)) &&
         "Expecting an extendable operand");

  if (MO.isExpr() || isExtended(MII.TSFlags)) {
    O << "#";
  } else if (MO.isImm()) {
    int ImmValue = MO.getImm();
    if (ImmValue < getMinValue(MII.TSFlags) ||
        ImmValue > getMaxValue(MII.TSFlags))
      O << "#";
  }
  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printUnsignedImmOperand(const MCInst *MI,
                                    unsigned OpNo, raw_ostream &O) const {
  O << MI->getOperand(OpNo).getImm();
}

void HexagonInstPrinter::printNegImmOperand(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) const {
  O << -MI->getOperand(OpNo).getImm();
}

void HexagonInstPrinter::printNOneImmOperand(const MCInst *MI, unsigned OpNo,
                                             raw_ostream &O) const {
  O << -1;
}

void HexagonInstPrinter::prints3_6ImmOperand(MCInst const *MI, unsigned OpNo,
                                             raw_ostream &O) const {
  int64_t Imm = MI->getOperand(OpNo).getImm();
  assert(((Imm & 0x3f) == 0) && "Lower 6 bits must be ZERO.");
  O << formatImm(Imm/64);
}

void HexagonInstPrinter::prints3_7ImmOperand(MCInst const *MI, unsigned OpNo,
                                             raw_ostream &O) const {
  int64_t Imm = MI->getOperand(OpNo).getImm();
  assert(((Imm & 0x7f) == 0) && "Lower 7 bits must be ZERO.");
  O << formatImm(Imm/128);
}

void HexagonInstPrinter::prints4_6ImmOperand(MCInst const *MI, unsigned OpNo,
                                             raw_ostream &O) const {
  int64_t Imm = MI->getOperand(OpNo).getImm();
  assert(((Imm & 0x3f) == 0) && "Lower 6 bits must be ZERO.");
  O << formatImm(Imm/64);
}

void HexagonInstPrinter::prints4_7ImmOperand(MCInst const *MI, unsigned OpNo,
                                             raw_ostream &O) const {
  int64_t Imm = MI->getOperand(OpNo).getImm();
  assert(((Imm & 0x7f) == 0) && "Lower 7 bits must be ZERO.");
  O << formatImm(Imm/128);
}

void HexagonInstPrinter::printMEMriOperand(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) const {
  const MCOperand& MO0 = MI->getOperand(OpNo);
  const MCOperand& MO1 = MI->getOperand(OpNo + 1);

  printRegName(O, MO0.getReg());
  O << " + #" << MO1.getImm();
}

void HexagonInstPrinter::printFrameIndexOperand(const MCInst *MI, unsigned OpNo,
                                                raw_ostream &O) const {
  const MCOperand& MO0 = MI->getOperand(OpNo);
  const MCOperand& MO1 = MI->getOperand(OpNo + 1);

  printRegName(O, MO0.getReg());
  O << ", #" << MO1.getImm();
}

void HexagonInstPrinter::printGlobalOperand(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) const {
  assert(MI->getOperand(OpNo).isExpr() && "Expecting expression");

  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printJumpTable(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) const {
  assert(MI->getOperand(OpNo).isExpr() && "Expecting expression");

  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printConstantPool(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) const {
  assert(MI->getOperand(OpNo).isExpr() && "Expecting expression");

  printOperand(MI, OpNo, O);
}

void HexagonInstPrinter::printBranchOperand(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) const {
  // Branches can take an immediate operand.  This is used by the branch
  // selection pass to print $+8, an eight byte displacement from the PC.
  llvm_unreachable("Unknown branch operand.");
}

void HexagonInstPrinter::printCallOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) const {
}

void HexagonInstPrinter::printAbsAddrOperand(const MCInst *MI, unsigned OpNo,
                                             raw_ostream &O) const {
}

void HexagonInstPrinter::printPredicateOperand(const MCInst *MI, unsigned OpNo,
                                               raw_ostream &O) const {
}

void HexagonInstPrinter::printSymbol(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O, bool hi) const {
  assert(MI->getOperand(OpNo).isImm() && "Unknown symbol operand");

  O << '#' << (hi ? "HI" : "LO") << "(#";
  printOperand(MI, OpNo, O);
  O << ')';
}

void HexagonInstPrinter::printExtBrtarget(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) const {
  const MCOperand &MO = MI->getOperand(OpNo);
  const MCInstrDesc &MII = getMII().get(MI->getOpcode());

  assert((isExtendable(MII.TSFlags) || isExtended(MII.TSFlags)) &&
         "Expecting an extendable operand");

  if (MO.isExpr() || isExtended(MII.TSFlags)) {
    O << "##";
  }
  printOperand(MI, OpNo, O);
}
