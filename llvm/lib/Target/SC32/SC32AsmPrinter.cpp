#include "SC32AsmPrinter.h"
#include "TargetInfo/SC32TargetInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

SC32AsmPrinter::SC32AsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> S)
    : AsmPrinter(TM, std::move(S)) {}

void SC32AsmPrinter::emitInstruction(const MachineInstr *MI) {
  MCInst Inst;

  Inst.setOpcode(MI->getOpcode());

  for (const MachineOperand &MO : MI->operands()) {
    switch (MO.getType()) {
    default:
      llvm_unreachable("unknown operand type");
    case MachineOperand::MO_Register:
      Inst.addOperand(MCOperand::createReg(MO.getReg()));
      break;
    case MachineOperand::MO_Immediate:
      Inst.addOperand(MCOperand::createImm(MO.getImm()));
      break;
    }
  }

  EmitToStreamer(*OutStreamer, Inst);
}

void SC32AsmPrinter::emitLinkage(const GlobalValue *GV, MCSymbol *GVSym) const {
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeSC32AsmPrinter() {
  RegisterAsmPrinter<SC32AsmPrinter>{getTheSC32Target()};
}
