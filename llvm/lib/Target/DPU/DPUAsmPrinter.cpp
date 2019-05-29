//===-- DPUAsmPrinter.cpp - DPU LLVM assembly writer ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the DPU assembly language.
//
//===----------------------------------------------------------------------===//

#include "DPUInstrInfo.h"
#include "DPUMCInstLower.h"
#include "DPUTargetMachine.h"
#include "InstPrinter/DPUInstPrinter.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace llvm {
extern Target TheDPUTarget;
}

namespace {
class DPUAsmPrinter : public AsmPrinter {
public:
  explicit DPUAsmPrinter(TargetMachine &TM,
                         std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  StringRef getPassName() const override { return "DPU Assembly Printer"; }

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &O) override;

  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                             unsigned AsmVariant, const char *ExtraCode,
                             raw_ostream &O) override;

  void printOperand(const MachineInstr *MI, int OpNum, raw_ostream &O,
                    const char *Modifier = nullptr);

  void EmitInstruction(const MachineInstr *MI) override;
};
} // namespace

bool DPUAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                    unsigned AsmVariant, const char *ExtraCode,
                                    raw_ostream &O) {
  printOperand(MI, OpNo, O, ExtraCode);
  return false;
}

bool DPUAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                                          unsigned AsmVariant,
                                          const char *ExtraCode,
                                          raw_ostream &O) {
  printOperand(MI, OpNo, O, ExtraCode);
  return false;
}

void DPUAsmPrinter::printOperand(const MachineInstr *MI, int OpNum,
                                 raw_ostream &O, const char *Modifier) {
  const MachineOperand &MO = MI->getOperand(OpNum);

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << DPUInstPrinter::getRegisterName(MO.getReg());
    break;

  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    break;

  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    break;

  case MachineOperand::MO_GlobalAddress:
    O << *getSymbol(MO.getGlobal());
    break;

  default:
    llvm_unreachable("<unknown operand type>");
  }
}

void DPUAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  DPUMCInstLower MCInstLowering(OutContext, *this);

  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);
  EmitToStreamer(*OutStreamer, TmpInst);
}

// Force static initialization.
extern "C" void LLVMInitializeDPUAsmPrinter() {
  RegisterAsmPrinter<DPUAsmPrinter> X(TheDPUTarget);
}
