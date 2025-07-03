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
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include <optional>

using namespace llvm;

void CFIFunctionFrameStreamer::updateReceiver(
    const std::optional<MCInst> &NewInst) {
  assert(!FrameIndices.empty() && hasUnfinishedDwarfFrameInfo() &&
         "FunctionUnitStreamer frame indices should be synced with "
         "MCStreamer's"); //! FIXME split this assertions and also check another
                          //! vectors
                          //! Add tests for nested frames

  auto Frames = getDwarfFrameInfos();
  assert(FrameIndices.back() < Frames.size());
  unsigned LastDirectiveIndex = FrameLastDirectiveIndices.back();
  unsigned CurrentDirectiveIndex =
      Frames[FrameIndices.back()].Instructions.size();
  assert(CurrentDirectiveIndex >= LastDirectiveIndex);

  const MCDwarfFrameInfo *LastFrame = &Frames[FrameIndices.back()];
  ArrayRef<MCCFIInstruction> Directives;
  if (LastDirectiveIndex < CurrentDirectiveIndex) {
    Directives = ArrayRef<MCCFIInstruction>(LastFrame->Instructions);
    Directives =
        Directives.drop_front(LastDirectiveIndex)
            .drop_back(LastFrame->Instructions.size() - CurrentDirectiveIndex);
  }

  auto MaybeLastInstruction = FrameLastInstructions.back();
  if (MaybeLastInstruction)
    Receiver->emitInstructionAndDirectives(*MaybeLastInstruction, Directives);
  else
    Receiver->startFunctionFrame(false /* TODO: should put isEH here */,
                                 Directives);

  FrameLastInstructions.back() = NewInst;
  FrameLastDirectiveIndices.back() = CurrentDirectiveIndex;
}

void CFIFunctionFrameStreamer::emitInstruction(const MCInst &Inst,
                                               const MCSubtargetInfo &STI) {
  if (hasUnfinishedDwarfFrameInfo()) {
    updateReceiver(Inst);
  }
}

void CFIFunctionFrameStreamer::emitCFIStartProcImpl(MCDwarfFrameInfo &Frame) {
  FrameLastInstructions.push_back(std::nullopt);
  FrameLastDirectiveIndices.push_back(0);
  FrameIndices.push_back(getNumFrameInfos());

  MCStreamer::emitCFIStartProcImpl(Frame);
}

void CFIFunctionFrameStreamer::emitCFIEndProcImpl(MCDwarfFrameInfo &CurFrame) {
  updateReceiver(std::nullopt);

  assert(!FrameIndices.empty() && "There should be at least one frame to pop");
  FrameLastDirectiveIndices.pop_back();
  FrameLastInstructions.pop_back();
  FrameIndices.pop_back();

  dbgs() << "finishing frame\n";
  Receiver->finishFunctionFrame();

  MCStreamer::emitCFIEndProcImpl(CurFrame);
}
