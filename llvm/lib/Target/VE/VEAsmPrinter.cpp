//===-- VEAsmPrinter.cpp - VE LLVM assembly writer ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format VE assembly language.
//
//===----------------------------------------------------------------------===//

#include "InstPrinter/VEInstPrinter.h"
#include "MCTargetDesc/VETargetStreamer.h"
#include "VE.h"
#include "VEInstrInfo.h"
#include "VETargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "ve-asmprinter"

namespace {
class VEAsmPrinter : public AsmPrinter {
  VETargetStreamer &getTargetStreamer() {
    return static_cast<VETargetStreamer &>(*OutStreamer->getTargetStreamer());
  }

public:
  explicit VEAsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  StringRef getPassName() const override { return "VE Assembly Printer"; }

  void EmitInstruction(const MachineInstr *MI) override;

  static const char *getRegisterName(unsigned RegNo) {
    return VEInstPrinter::getRegisterName(RegNo);
  }
};
} // end of anonymous namespace

void VEAsmPrinter::EmitInstruction(const MachineInstr *MI) {

  switch (MI->getOpcode()) {
  default:
    break;
  case TargetOpcode::DBG_VALUE:
    // FIXME: Debug Value.
    return;
  }
  MachineBasicBlock::const_instr_iterator I = MI->getIterator();
  MachineBasicBlock::const_instr_iterator E = MI->getParent()->instr_end();
  do {
    MCInst TmpInst;
    LowerVEMachineInstrToMCInst(&*I, TmpInst, *this);
    EmitToStreamer(*OutStreamer, TmpInst);
  } while ((++I != E) && I->isInsideBundle()); // Delay slot check.
}

// Force static initialization.
extern "C" void LLVMInitializeVEAsmPrinter() {
  RegisterAsmPrinter<VEAsmPrinter> X(getTheVETarget());
}
