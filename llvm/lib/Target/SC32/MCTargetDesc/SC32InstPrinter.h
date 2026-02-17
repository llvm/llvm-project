#ifndef LLVM_LIB_TARGET_SC32_MCTARGETDESC_SC32INSTPRINTER_H
#define LLVM_LIB_TARGET_SC32_MCTARGETDESC_SC32INSTPRINTER_H

#include "llvm/MC/MCInstPrinter.h"

namespace llvm {

class SC32InstPrinter : public MCInstPrinter {
public:
  using MCInstPrinter::MCInstPrinter;

  std::pair<const char *, uint64_t>
  getMnemonic(const MCInst &MI) const override;

  void printInst(const MCInst *MI, uint64_t Address, StringRef Annot,
                 const MCSubtargetInfo &STI, raw_ostream &OS) override;

  void printOperand(const MCInst *MI, unsigned i, raw_ostream &OS);

  void printInstruction(const MCInst *MI, uint64_t Address, raw_ostream &OS);

  const char *getRegisterName(MCRegister Reg);
};

} // namespace llvm

#endif
