//===-- M88kAsmPrinter.cpp - M88k LLVM assembly writer ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format M88k assembly language.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/M88kInstPrinter.h"
//#include "MCTargetDesc/M88kMCExpr.h"
//#include "MCTargetDesc/M88kTargetStreamer.h"
#include "M88k.h"
#include "M88kInstrInfo.h"
#include "M88kMCInstLower.h"
#include "M88kTargetMachine.h"
#include "TargetInfo/M88kTargetInfo.h"
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
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

// TODO:
// %hi16() and %lo16() for addresses

namespace {
class M88kAsmPrinter : public AsmPrinter {
#if 0
    M88kTargetStreamer &getTargetStreamer() {
      return static_cast<M88kTargetStreamer &>(
          *OutStreamer->getTargetStreamer());
    }
#endif
public:
  explicit M88kAsmPrinter(TargetMachine &TM,
                          std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)) {}

  StringRef getPassName() const override { return "M88k Assembly Printer"; }

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &OS) override;
  void emitInstruction(const MachineInstr *MI) override;
};
} // end of anonymous namespace

bool M88kAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                     const char *ExtraCode, raw_ostream &OS) {
  if (ExtraCode)
    return AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, OS);
  M88kMCInstLower Lower(MF->getContext(), *this);
  MCOperand MO(Lower.lowerOperand(MI->getOperand(OpNo),
                                  MF->getSubtarget().getRegisterInfo()));
  M88kInstPrinter::printOperand(MO, MAI, OS);
  return false;
}

void M88kAsmPrinter::emitInstruction(const MachineInstr *MI) {
  MachineBasicBlock::const_instr_iterator I = MI->getIterator();
  MachineBasicBlock::const_instr_iterator E = MI->getParent()->instr_end();

  do {
    // Skip the BUNDLE pseudo instruction and lower the contents.
    if (I->isBundle())
      continue;

    MCInst LoweredMI;
    switch (I->getOpcode()) {
    case M88k::RET:
      LoweredMI = MCInstBuilder(M88k::JMP).addReg(M88k::R1);
      break;

    case M88k::RETn:
      LoweredMI = MCInstBuilder(M88k::JMPn).addReg(M88k::R1);
      break;

    default:
      M88kMCInstLower Lower(MF->getContext(), *this);
      Lower.lower(&*I, LoweredMI);
      // doLowerInstr(MI, LoweredMI);
      break;
    }
    EmitToStreamer(*OutStreamer, LoweredMI);
  } while ((++I != E) && I->isInsideBundle()); // Delay slot check.
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeM88kAsmPrinter() {
  RegisterAsmPrinter<M88kAsmPrinter> X(getTheM88kTarget());
}
