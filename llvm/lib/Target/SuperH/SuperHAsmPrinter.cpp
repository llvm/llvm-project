//===-- SuperHAsmPrinter.cpp - SH LLVM Assembly Printer ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format SuperH assembly language.
//
//===-----------------------------------------------------------------------===//

#include "MCTargetDesc/SuperHInstPrinter.h"
#include "MCTargetDesc/SuperHMCAsmInfo.h"
#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "MCTargetDesc/SuperHTargetStreamer.h"
#include "SuperH.h"
#include "SuperHMCInstLower.h"
#include "TargetInfo/SuperHTargetInfo.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {

class SuperHAsmPrinter : public AsmPrinter {
	SuperHTargetStreamer &getTargetStreamer() {
		return static_cast<SuperHTargetStreamer &>(
			*OutStreamer->getTargetStreamer());
	}

public:
  explicit SuperHAsmPrinter(TargetMachine &TM,
                           std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer), ID) {}

  StringRef getPassName() const override { return "SuperH Assembly Printer"; }

  void printOperand(const MachineInstr *MI, int opNum, raw_ostream &OS);
  void printMemOperand(const MachineInstr *MI, int opNum, raw_ostream &OS);

  void emitFunctionBodyStart() override;
  void emitInstruction(const MachineInstr *MI) override;

  static const char *getRegisterName(MCRegister Reg) {
    return SuperHInstPrinter::getRegisterName(Reg);
  }

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             const char *ExtraCode, raw_ostream &O) override;

  static char ID;
};

} // namespace


void SuperHAsmPrinter::emitFunctionBodyStart() {
  AsmPrinter::emitFunctionBodyStart();
}

void SuperHAsmPrinter::emitInstruction(const MachineInstr *MI) {
  SuperHMCInstLower MCInstLowering(OutContext, *this);

  MCInst I;
  MCInstLowering.lowerInstruction(*MI, I);
  EmitToStreamer(*OutStreamer, I);
}

bool SuperHAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) {
  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, O))
    return false;


	return false;
}

bool SuperHAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) {
	return AsmPrinter::PrintAsmMemoryOperand(MI, OpNo, ExtraCode, O);
}

char SuperHAsmPrinter::ID = 0;
INITIALIZE_PASS(SuperHAsmPrinter, "sh-asm-printer", "SuperH Assembly Printer", false, false)

// Force static initialization.
extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeSuperHAsmPrinter() {
  RegisterAsmPrinter<SuperHAsmPrinter> X(getTheSuperHTarget());
  RegisterAsmPrinter<SuperHAsmPrinter> Y(getTheSuperHLETarget());
}
