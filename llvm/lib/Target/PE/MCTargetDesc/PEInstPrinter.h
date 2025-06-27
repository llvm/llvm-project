/* --- PEInstPrinter.h --- */

/* ------------------------------------------
Author: 高宇翔
Date: 4/9/2025
------------------------------------------ */

#ifndef PEINSTPRINTER_H
#define PEINSTPRINTER_H

#include "MCTargetDesc/PEMCTargetDesc.h"
#include "llvm/MC/MCInstPrinter.h"

namespace llvm {
class PEInstPrinter : public MCInstPrinter {
public:
  PEInstPrinter(const MCAsmInfo &MAI, const MCInstrInfo &MII,
                const MCRegisterInfo &MRI)
      : MCInstPrinter(MAI, MII, MRI) {}

  void printCustomAliasOperand(const MCInst *MI, uint64_t Address,
                               unsigned OpIdx, unsigned PrintMethodIdx,
                               raw_ostream &OS);
  void printBranchOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O);
  std::pair<const char *, uint64_t>
  getMnemonic(const MCInst &MI) const override;
  void printInstruction(const MCInst *MI, uint64_t Address, raw_ostream &O);
  static const char *getRegisterName(MCRegister Reg);

  void printRegName(raw_ostream &OS, MCRegister Reg) override;
  void printInst(const MCInst *MI, uint64_t Address, StringRef Annot,
                 const MCSubtargetInfo &STI, raw_ostream &O) override;
  void printCCOperand(const MCInst *MI, int OpNum, raw_ostream &O);
  void printU6(const MCInst *MI, int OpNum, raw_ostream &O);

  bool printAliasInstr(const MCInst *MI, uint64_t Address, raw_ostream &OS);

  void PrintMemOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O);
  void PrintVMemOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O);
  void PrintPtrOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O);

private:
  void printMemOperandRI(const MCInst *MI, unsigned OpNum, raw_ostream &O);
  void printOperand(const MCInst *MI, unsigned OpNum, raw_ostream &O);
  void printOperand(const MCInst *MI, uint64_t /*Address*/, unsigned OpNum,
                    raw_ostream &O) {
    printOperand(MI, OpNum, O);
  }
  void printPredicateOperand(const MCInst *MI, unsigned OpNum, raw_ostream &O);
  void printBRCCPredicateOperand(const MCInst *MI, unsigned OpNum,
                                 raw_ostream &O);
  void printU6ShiftedBy(unsigned ShiftBy, const MCInst *MI, int OpNum,
                        raw_ostream &O);
};
} // namespace llvm

#endif // PEINSTPRINTER_H