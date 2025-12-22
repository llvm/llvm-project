//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFCFIChecker/DWARFCFIFunctionFrameAnalyzer.h"

using namespace llvm;

CFIFunctionFrameAnalyzer::~CFIFunctionFrameAnalyzer() {
  assert(UIAs.empty() &&
         "all frames should be closed before the analysis finishes");
}

void CFIFunctionFrameAnalyzer::startFunctionFrame(
    bool IsEH, ArrayRef<MCCFIInstruction> Prologue) {
  UIAs.emplace_back(&getContext(), MCII, IsEH, Prologue);
}

void CFIFunctionFrameAnalyzer::emitInstructionAndDirectives(
    const MCInst &Inst, ArrayRef<MCCFIInstruction> Directives) {
  assert(!UIAs.empty() && "if the instruction is in a frame, there should be "
                          "a analysis instantiated for it");
  UIAs.back().update(Inst, Directives);
}

void CFIFunctionFrameAnalyzer::finishFunctionFrame() {
  assert(!UIAs.empty() && "there should be an analysis for each frame");
  UIAs.pop_back();
}
