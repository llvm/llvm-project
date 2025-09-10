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
  assert(hasUnfinishedDwarfFrameInfo() &&
         "should have an unfinished DWARF frame here");
  assert(!FrameIndices.empty() &&
         "there should be an index available for the current frame");
  assert(FrameIndices.size() == LastInstructions.size());
  assert(LastInstructions.size() == LastDirectiveIndices.size());

  auto Frames = getDwarfFrameInfos();
  assert(FrameIndices.back() < Frames.size());
  unsigned LastDirectiveIndex = LastDirectiveIndices.back();
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

  auto MaybeLastInstruction = LastInstructions.back();
  if (MaybeLastInstruction)
    // The directives are associated with an instruction.
    Receiver->emitInstructionAndDirectives(*MaybeLastInstruction, Directives);
  else
    // The directives are the prologue directives.
    Receiver->startFunctionFrame(false /* TODO: should put isEH here */,
                                 Directives);

  // Update the internal state for the top frame.
  LastInstructions.back() = NewInst;
  LastDirectiveIndices.back() = CurrentDirectiveIndex;
}

void CFIFunctionFrameStreamer::emitInstruction(const MCInst &Inst,
                                               const MCSubtargetInfo &STI) {
  if (hasUnfinishedDwarfFrameInfo())
    // Send the last instruction with the unsent directives already in the frame
    // to the receiver.
    updateReceiver(Inst);
}

void CFIFunctionFrameStreamer::emitCFIStartProcImpl(MCDwarfFrameInfo &Frame) {
  LastInstructions.push_back(std::nullopt);
  LastDirectiveIndices.push_back(0);
  FrameIndices.push_back(getNumFrameInfos());

  MCStreamer::emitCFIStartProcImpl(Frame);
}

void CFIFunctionFrameStreamer::emitCFIEndProcImpl(MCDwarfFrameInfo &CurFrame) {
  // Send the last instruction with the final directives of the current frame to
  // the receiver.
  updateReceiver(std::nullopt);

  assert(!FrameIndices.empty() && "There should be at least one frame to pop");
  LastDirectiveIndices.pop_back();
  LastInstructions.pop_back();
  FrameIndices.pop_back();

  Receiver->finishFunctionFrame();

  MCStreamer::emitCFIEndProcImpl(CurFrame);
}
