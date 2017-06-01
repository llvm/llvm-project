//===-- HexagonInstPrinter.h - Convert Hexagon MCInst to assembly syntax --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_INSTPRINTER_HEXAGONINSTPRINTER_H
#define LLVM_LIB_TARGET_HEXAGON_INSTPRINTER_HEXAGONINSTPRINTER_H

#include "llvm/MC/MCInstPrinter.h"

namespace llvm {
/// Prints bundles as a newline separated list of individual instructions
/// Duplexes are separated by a vertical tab \v character
/// A trailing line includes bundle properties such as endloop0/1
///
/// r0 = add(r1, r2)
/// r0 = #0 \v jump 0x0
/// :endloop0 :endloop1
class HexagonInstPrinter : public MCInstPrinter {
public:
  explicit HexagonInstPrinter(MCAsmInfo const &MAI, MCInstrInfo const &MII,
                              MCRegisterInfo const &MRI);
  void printInst(MCInst const *MI, raw_ostream &O, StringRef Annot,
                 const MCSubtargetInfo &STI) override;
  virtual StringRef getOpcodeName(unsigned Opcode) const;
  void printInstruction(MCInst const *MI, raw_ostream &O);

  StringRef getRegName(unsigned RegNo) const;
  static char const *getRegisterName(unsigned RegNo);
  void printRegName(raw_ostream &O, unsigned RegNo) const override;

  void printOperand(MCInst const *MI, unsigned OpNo, raw_ostream &O) const;
  void printExtOperand(MCInst const *MI, unsigned OpNo, raw_ostream &O) const;
  void printUnsignedImmOperand(MCInst const *MI, unsigned OpNo,
                               raw_ostream &O) const;
  void printNegImmOperand(MCInst const *MI, unsigned OpNo,
                          raw_ostream &O) const;
  void printNOneImmOperand(MCInst const *MI, unsigned OpNo,
                           raw_ostream &O) const;
  void prints3_6ImmOperand(MCInst const *MI, unsigned OpNo,
                           raw_ostream &O) const;
  void prints3_7ImmOperand(MCInst const *MI, unsigned OpNo,
                           raw_ostream &O) const;
  void prints4_6ImmOperand(MCInst const *MI, unsigned OpNo,
                           raw_ostream &O) const;
  void prints4_7ImmOperand(MCInst const *MI, unsigned OpNo,
                           raw_ostream &O) const;
  void printBranchOperand(MCInst const *MI, unsigned OpNo,
                          raw_ostream &O) const;
  void printCallOperand(MCInst const *MI, unsigned OpNo, raw_ostream &O) const;
  void printAbsAddrOperand(MCInst const *MI, unsigned OpNo,
                           raw_ostream &O) const;
  void printPredicateOperand(MCInst const *MI, unsigned OpNo,
                             raw_ostream &O) const;
  void printGlobalOperand(MCInst const *MI, unsigned OpNo,
                          raw_ostream &O) const;
  void printJumpTable(MCInst const *MI, unsigned OpNo, raw_ostream &O) const;
  void printBrtarget(MCInst const *MI, unsigned OpNo, raw_ostream &O) const;

  void printConstantPool(MCInst const *MI, unsigned OpNo, raw_ostream &O) const;

  void printSymbolHi(MCInst const *MI, unsigned OpNo, raw_ostream &O) const {
    printSymbol(MI, OpNo, O, true);
  }
  void printSymbolLo(MCInst const *MI, unsigned OpNo, raw_ostream &O) const {
    printSymbol(MI, OpNo, O, false);
  }

  MCAsmInfo const &getMAI() const { return MAI; }
  MCInstrInfo const &getMII() const { return MII; }

protected:
  void printSymbol(MCInst const *MI, unsigned OpNo, raw_ostream &O,
                   bool hi) const;

private:
  MCInstrInfo const &MII;

  bool HasExtender;
  void setExtender(MCInst const &MCI);
};

} // end namespace llvm

#endif
