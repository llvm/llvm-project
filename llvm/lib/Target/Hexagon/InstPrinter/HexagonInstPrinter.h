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

#ifndef HEXAGONINSTPRINTER_H
#define HEXAGONINSTPRINTER_H

#include "HexagonMCInst.h"
#include "llvm/MC/MCInstPrinter.h"

namespace llvm {
  class HexagonInstPrinter : public MCInstPrinter {
  public:
    explicit HexagonInstPrinter(const MCAsmInfo &MAI,
                                const MCInstrInfo &MII,
                                const MCRegisterInfo &MRI)
      : MCInstPrinter(MAI, MII, MRI) {}

    virtual void printInst(const MCInst *MI, raw_ostream &O, StringRef Annot);
    void printInst(const HexagonMCInst *MI, raw_ostream &O, StringRef Annot);
    virtual StringRef getOpcodeName(unsigned Opcode) const;
    void printInstruction(const MCInst *MI, raw_ostream &O);
    StringRef getRegName(unsigned RegNo) const;
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

    void printConstantPool(const MCInst *MI, unsigned OpNo,
                           raw_ostream &O) const;

    void printSymbolHi(const MCInst *MI, unsigned OpNo, raw_ostream &O) const
      { printSymbol(MI, OpNo, O, true); }
    void printSymbolLo(const MCInst *MI, unsigned OpNo, raw_ostream &O) const
      { printSymbol(MI, OpNo, O, false); }

    bool isConstExtended(const MCInst *MI) const;
  protected:
    void printSymbol(const MCInst *MI, unsigned OpNo, raw_ostream &O, bool hi)
           const;
  };

} // end namespace llvm

#endif
