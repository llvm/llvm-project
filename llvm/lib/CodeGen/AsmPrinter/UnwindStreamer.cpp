//===- CodeGen/AsmPrinter/UnwindStreamer.cpp - Unwind Directive Streamer --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing unwind info into assembly files.
//
//===----------------------------------------------------------------------===//

#include "UnwindStreamer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

namespace llvm {
class MachineFunction;
} // end namespace llvm

UnwindStreamer::UnwindStreamer(AsmPrinter *A) : Asm(A) {}

UnwindStreamer::~UnwindStreamer() = default;

void UnwindStreamer::beginFunction(const MachineFunction *MF) {
  assert(Asm->needsCFIMoves() == AsmPrinter::CFI_M_Debug);
  if (!HasEmittedDebugFrame) {
    Asm->OutStreamer->emitCFISections(false, true);
    HasEmittedDebugFrame = true;
  }
  Asm->OutStreamer->emitCFIStartProc(/*IsSimple=*/false);
}

void UnwindStreamer::endFunction(const MachineFunction *MF) {
  Asm->OutStreamer->emitCFIEndProc();
}
