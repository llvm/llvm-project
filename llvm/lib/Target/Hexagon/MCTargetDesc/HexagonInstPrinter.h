//===-- HexagonInstPrinter.h - Convert Hexagon MCInst to assembly syntax --===//
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

#ifndef LLVM_LIB_TARGET_HEXAGON_INSTPRINTER_HEXAGONINSTPRINTER_H
#define LLVM_LIB_TARGET_HEXAGON_INSTPRINTER_HEXAGONINSTPRINTER_H

#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"

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
    explicit HexagonInstPrinter(MCAsmInfo const &MAI,
                                MCInstrInfo const &MII,
                                MCRegisterInfo const &MRI)
      : MCInstPrinter(MAI, MII, MRI), MII(MII) {}

    void printInst(MCInst const *MI, raw_ostream &O, StringRef Annot,
                   const MCSubtargetInfo &STI) override;
    virtual StringRef getOpcodeName(unsigned Opcode) const;
    void printInstruction(const MCInst *MI, raw_ostream &O);
    void printRegName(raw_ostream &OS, unsigned RegNo) const override;
    static const char *getRegisterName(unsigned RegNo);

    void printOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O) const;
    void printImmOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O) const;
    void printExtOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O) const;
    void printUnsignedImmOperand(const MCInst *MI, unsigned OpNo,
                                 raw_ostream &O) const;
    void printNegImmOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printNOneImmOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
             const;
    void prints3_6ImmOperand(MCInst const *MI, unsigned OpNo,
                             raw_ostream &O) const;
    void prints3_7ImmOperand(MCInst const *MI, unsigned OpNo,
                             raw_ostream &O) const;
    void prints4_6ImmOperand(MCInst const *MI, unsigned OpNo,
                             raw_ostream &O) const;
    void prints4_7ImmOperand(MCInst const *MI, unsigned OpNo,
                             raw_ostream &O) const;
    void printMEMriOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printFrameIndexOperand(const MCInst *MI, unsigned OpNo,
                                raw_ostream &O) const;
    void printBranchOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printCallOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printAbsAddrOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printPredicateOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printGlobalOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O)
           const;
    void printJumpTable(const MCInst *MI, unsigned OpNo, raw_ostream &O) const;
    void printExtBrtarget(const MCInst *MI, unsigned OpNo, raw_ostream &O) const;

    void printConstantPool(const MCInst *MI, unsigned OpNo,
                           raw_ostream &O) const;

    void printSymbolHi(const MCInst *MI, unsigned OpNo, raw_ostream &O) const
      { printSymbol(MI, OpNo, O, true); }
    void printSymbolLo(const MCInst *MI, unsigned OpNo, raw_ostream &O) const
      { printSymbol(MI, OpNo, O, false); }

    const MCInstrInfo &getMII() const {
      return MII;
    }

  protected:
    void printSymbol(const MCInst *MI, unsigned OpNo, raw_ostream &O, bool hi)
           const;

  private:
    const MCInstrInfo &MII;

    bool HasExtender;
    void setExtender(MCInst const &MCI);
  };

} // end namespace llvm

#endif
