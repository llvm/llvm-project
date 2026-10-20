//===-- Toy64InstPrinter.h - Toy64 MCInst to assembly ----------*- C++ -*-===//

#ifndef LLVM_LIB_TARGET_TOY64_MCTARGETDESC_TOY64INSTPRINTER_H
#define LLVM_LIB_TARGET_TOY64_MCTARGETDESC_TOY64INSTPRINTER_H

#include "llvm/MC/MCInstPrinter.h"

namespace llvm {

class Toy64InstPrinter : public MCInstPrinter {
public:
  Toy64InstPrinter(
      const MCAsmInfo &MAI,
      const MCInstrInfo &MII,
      const MCRegisterInfo &MRI)
      : MCInstPrinter(MAI, MII, MRI) {}

  void printInst(
      const MCInst *MI,
      uint64_t Address,
      StringRef Annot,
      const MCSubtargetInfo &STI,
      raw_ostream &OS) override;

  void printRegName(
      raw_ostream &OS,
      MCRegister Reg) override;

  std::pair<const char *, uint64_t>
  getMnemonic(const MCInst &MI) const override;

private:
  void printOperand(
      const MCInst *MI,
      unsigned OpNo,
      raw_ostream &OS);
};

} // namespace llvm

#endif