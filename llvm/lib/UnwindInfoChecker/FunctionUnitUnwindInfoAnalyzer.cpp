//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/UnwindInfoChecker/FunctionUnitUnwindInfoAnalyzer.h"

using namespace llvm;

void CFIFunctionFrameAnalyzer::startFunctionUnit(
    bool IsEH, ArrayRef<MCCFIInstruction> Prologue) {
  UIAs.emplace_back(&getContext(), MCII, IsEH, Prologue);
}

void CFIFunctionFrameAnalyzer::emitInstructionAndDirectives(
    const MCInst &Inst, ArrayRef<MCCFIInstruction> Directives) {
  assert(!UIAs.empty() && "If the instruction is in a frame, there should be "
                          "a analysis instantiated for it");
  UIAs.back().update(Inst, Directives);
}

void CFIFunctionFrameAnalyzer::finishFunctionUnit() {
  assert(!UIAs.empty() && "There should be an analysis for each frame");
  UIAs.pop_back();
}
