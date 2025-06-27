//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFCFIChecker/DWARFCFIFunctionFrameStreamer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include <cstdio>
#include <optional>

using namespace llvm;

std::pair<unsigned, unsigned>
CFIFunctionFrameStreamer::updateDirectivesRange() {
  auto Frames = getDwarfFrameInfos();
  unsigned CurrentCFIDirectiveIndex = 0;
  if (hasUnfinishedDwarfFrameInfo()) {
    assert(!FrameIndices.empty() && "FunctionUnitStreamer frame indices should "
                                    "be synced with MCStreamer's");
    assert(FrameIndices.back() < Frames.size());
    CurrentCFIDirectiveIndex = Frames[FrameIndices.back()].Instructions.size();
  }

  assert(CurrentCFIDirectiveIndex >= LastDirectiveIndex);
  std::pair<unsigned, unsigned> CFIDirectivesRange(LastDirectiveIndex,
                                                   CurrentCFIDirectiveIndex);
  LastDirectiveIndex = CurrentCFIDirectiveIndex;
  return CFIDirectivesRange;
}

void CFIFunctionFrameStreamer::updateAnalyzer() {
  if (FrameIndices.empty()) {
    auto CFIDirectivesRange = updateDirectivesRange();
    assert(CFIDirectivesRange.first == CFIDirectivesRange.second &&
           "CFI directives should be in some frame");
    return;
  }

  const MCDwarfFrameInfo *LastFrame =
      &getDwarfFrameInfos()[FrameIndices.back()];

  auto DirectivesRange = updateDirectivesRange();
  ArrayRef<MCCFIInstruction> Directives;
  if (DirectivesRange.first < DirectivesRange.second) {
    Directives = ArrayRef<MCCFIInstruction>(LastFrame->Instructions);
    Directives =
        Directives.drop_front(DirectivesRange.first)
            .drop_back(LastFrame->Instructions.size() - DirectivesRange.second);
  }

  if (LastInstruction) {
    Analyzer->emitInstructionAndDirectives(LastInstruction.value(), Directives);
  } else {
    Analyzer->startFunctionUnit(false /* TODO: should put isEH here */,
                                Directives);
  }
}

void CFIFunctionFrameStreamer::emitInstruction(const MCInst &Inst,
                                               const MCSubtargetInfo &STI) {
  updateAnalyzer();
  LastInstruction = Inst;
}

void CFIFunctionFrameStreamer::emitCFIStartProcImpl(MCDwarfFrameInfo &Frame) {
  updateAnalyzer();
  FrameIndices.push_back(getNumFrameInfos());
  LastInstruction = std::nullopt;
  LastDirectiveIndex = 0;
  MCStreamer::emitCFIStartProcImpl(Frame);
}

void CFIFunctionFrameStreamer::emitCFIEndProcImpl(MCDwarfFrameInfo &CurFrame) {
  updateAnalyzer();

  assert(!FrameIndices.empty() && "There should be at least one frame to pop");
  FrameIndices.pop_back();

  Analyzer->finishFunctionUnit();

  LastInstruction = std::nullopt;
  LastDirectiveIndex = 0;
  if (!FrameIndices.empty()) {
    auto DwarfFrameInfos = getDwarfFrameInfos();
    assert(FrameIndices.back() < DwarfFrameInfos.size());

    LastDirectiveIndex =
        DwarfFrameInfos[FrameIndices.back()].Instructions.size();
  }
  MCStreamer::emitCFIEndProcImpl(CurFrame);
}
