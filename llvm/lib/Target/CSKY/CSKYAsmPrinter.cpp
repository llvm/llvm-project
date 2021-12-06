//===-- CSKYAsmPrinter.cpp - CSKY LLVM assembly writer --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the CSKY assembly language.
//
//===----------------------------------------------------------------------===//
#include "CSKYAsmPrinter.h"
#include "CSKY.h"
#include "CSKYTargetMachine.h"
#include "MCTargetDesc/CSKYInstPrinter.h"
#include "MCTargetDesc/CSKYMCExpr.h"
#include "TargetInfo/CSKYTargetInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "csky-asm-printer"

STATISTIC(CSKYNumInstrsCompressed,
          "Number of C-SKY Compressed instructions emitted");

CSKYAsmPrinter::CSKYAsmPrinter(llvm::TargetMachine &TM,
                               std::unique_ptr<llvm::MCStreamer> Streamer)
    : AsmPrinter(TM, std::move(Streamer)), MCInstLowering(OutContext, *this) {}

bool CSKYAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<CSKYSubtarget>();
  return AsmPrinter::runOnMachineFunction(MF);
}

#define GEN_COMPRESS_INSTR
#include "CSKYGenCompressInstEmitter.inc"
void CSKYAsmPrinter::EmitToStreamer(MCStreamer &S, const MCInst &Inst) {
  MCInst CInst;
  bool Res = compressInst(CInst, Inst, *Subtarget, OutStreamer->getContext());
  if (Res)
    ++CSKYNumInstrsCompressed;
  AsmPrinter::EmitToStreamer(*OutStreamer, Res ? CInst : Inst);
}

// Simple pseudo-instructions have their lowering (with expansion to real
// instructions) auto-generated.
#include "CSKYGenMCPseudoLowering.inc"

void CSKYAsmPrinter::emitInstruction(const MachineInstr *MI) {
  // Do any auto-generated pseudo lowerings.
  if (emitPseudoExpansionLowering(*OutStreamer, MI))
    return;

  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);
  EmitToStreamer(*OutStreamer, TmpInst);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeCSKYAsmPrinter() {
  RegisterAsmPrinter<CSKYAsmPrinter> X(getTheCSKYTarget());
}
