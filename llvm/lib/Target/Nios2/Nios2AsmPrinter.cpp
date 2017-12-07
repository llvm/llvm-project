//===-- Nios2AsmPrinter.cpp - Nios2 LLVM Assembly Printer -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format NIOS2 assembly language.
//
//===----------------------------------------------------------------------===//

#include "InstPrinter/Nios2InstPrinter.h"
#include "MCTargetDesc/Nios2BaseInfo.h"
#include "Nios2.h"
#include "Nios2TargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "nios2-asm-printer"

namespace {

class Nios2AsmPrinter : public AsmPrinter {

public:
  explicit Nios2AsmPrinter(TargetMachine &TM,
                           std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  StringRef getPassName() const override { return "Nios2 Assembly Printer"; }

  //- EmitInstruction() must exists or will have run time error.
  void EmitInstruction(const MachineInstr *MI) override;
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &O) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                             unsigned AsmVariant, const char *ExtraCode,
                             raw_ostream &O) override;
  void printOperand(const MachineInstr *MI, int opNum, raw_ostream &O);
  void EmitFunctionEntryLabel() override;
};
} // namespace

//- EmitInstruction() must exists or will have run time error.
void Nios2AsmPrinter::EmitInstruction(const MachineInstr *MI) {

  //  Print out both ordinary instruction and boudle instruction
  MachineBasicBlock::const_instr_iterator I = MI->getIterator();
  MachineBasicBlock::const_instr_iterator E = MI->getParent()->instr_end();

  do {

    if (I->isPseudo()) {
      llvm_unreachable("Pseudo opcode found in EmitInstruction()");
    }

    MCInst TmpInst0;
    LowerNios2MachineInstToMCInst(&*I, TmpInst0, *this);
    EmitToStreamer(*OutStreamer, TmpInst0);
  } while ((++I != E) && I->isInsideBundle()); // Delay slot check
}

//		.type	main,@function
//->		.ent	main                    # @main
//	main:
void Nios2AsmPrinter::EmitFunctionEntryLabel() {
  OutStreamer->EmitLabel(CurrentFnSym);
}

// Print out an operand for an inline asm expression.
bool Nios2AsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                                      unsigned AsmVariant,
                                      const char *ExtraCode, raw_ostream &O) {
  printOperand(MI, OpNum, O);
  return false;
}

bool Nios2AsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                            unsigned OpNum, unsigned AsmVariant,
                                            const char *ExtraCode,
                                            raw_ostream &O) {
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier

  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isReg() && "unexpected inline asm memory operand");
  O << "($" << Nios2InstPrinter::getRegisterName(MO.getReg()) << ")";

  return false;
}

void Nios2AsmPrinter::printOperand(const MachineInstr *MI, int opNum,
                                   raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(opNum);
  bool closeP = false;

  if (MO.getTargetFlags())
    closeP = true;

  switch (MO.getTargetFlags()) {
  case Nios2FG::MO_ABS_HI:
    O << "%hiadj(";
    break;
  case Nios2FG::MO_ABS_LO:
    O << "%lo(";
    break;
  }

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << '$'
      << StringRef(Nios2InstPrinter::getRegisterName(MO.getReg())).lower();
    break;

  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    break;

  case MachineOperand::MO_MachineBasicBlock:
    MO.getMBB()->getSymbol()->print(O, MAI);
    return;

  case MachineOperand::MO_GlobalAddress:
    getSymbol(MO.getGlobal())->print(O, MAI);
    break;

  case MachineOperand::MO_BlockAddress:
    O << GetBlockAddressSymbol(MO.getBlockAddress())->getName();
    break;

  case MachineOperand::MO_ExternalSymbol:
    O << MO.getSymbolName();
    break;

  default:
    llvm_unreachable("<unknown operand type>");
  }

  if (closeP)
    O << ")";
}

// Force static initialization.
extern "C" void LLVMInitializeNios2AsmPrinter() {
  RegisterAsmPrinter<Nios2AsmPrinter> X(getTheNios2Target());
}
