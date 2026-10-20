//===-- Toy64InstPrinter.cpp - Toy64 MCInst to assembly -------------------===//

#include "Toy64InstPrinter.h"
#include "Toy64MCTargetDesc.h"

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

std::pair<const char *, uint64_t>
Toy64InstPrinter::getMnemonic(const MCInst &MI) const {
  switch (MI.getOpcode()) {
  case Toy64::ADD:
    return {"add", 0};

  case Toy64::SUB:
    return {"sub", 0};

  case Toy64::ADDI:
    return {"addi", 0};

  case Toy64::RET:
    return {"ret", 0};

  default:
    llvm_unreachable(
        "unknown Toy64 opcode in instruction printer");
  }
}

void Toy64InstPrinter::printRegName(
    raw_ostream &OS,
    MCRegister Reg) {
  OS << MRI.getName(Reg);
}

void Toy64InstPrinter::printOperand(
    const MCInst *MI,
    unsigned OpNo,
    raw_ostream &OS) {

  const MCOperand &Op = MI->getOperand(OpNo);

  if (Op.isReg()) {
    printRegName(OS, Op.getReg());
    return;
  }

  if (Op.isImm()) {
    OS << Op.getImm();
    return;
  }

  llvm_unreachable(
      "unsupported Toy64 MC operand");
}

void Toy64InstPrinter::printInst(
    const MCInst *MI,
    uint64_t Address,
    StringRef Annot,
    const MCSubtargetInfo &STI,
    raw_ostream &OS) {

  auto [Mnemonic, Bits] =
      getMnemonic(*MI);

  OS << '\t' << Mnemonic;

  for (unsigned I = 0;
       I < MI->getNumOperands();
       ++I) {

    if (I == 0)
      OS << '\t';
    else
      OS << ", ";

    printOperand(MI, I, OS);
  }

  printAnnotation(OS, Annot);
}