#include "SC32InstPrinter.h"
#include "llvm/MC/MCInst.h"

using namespace llvm;

#include "SC32GenAsmWriter.inc"

void SC32InstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                StringRef Annot, const MCSubtargetInfo &STI,
                                raw_ostream &OS) {
  printInstruction(MI, Address, OS);
}

void SC32InstPrinter::printOperand(const MCInst *MI, unsigned i,
                                   raw_ostream &OS) {
  const MCOperand &MO = MI->getOperand(i);

  if (MO.isReg()) {
    OS << getRegisterName(MO.getReg());
  } else if (MO.isImm()) {
    OS << '#' << MO.getImm();
  } else {
    llvm_unreachable("unknwon operand type");
  }
}
