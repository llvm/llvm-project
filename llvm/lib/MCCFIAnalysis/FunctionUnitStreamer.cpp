
#include "llvm/MCCFIAnalysis/FunctionUnitStreamer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include <cstdio>
#include <optional>

using namespace llvm;

std::pair<unsigned, unsigned> FunctionUnitStreamer::updateDirectivesRange() {
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

void FunctionUnitStreamer::updateUIA() {
  if (FrameIndices.empty()) {
    auto CFIDirectivesRange = updateDirectivesRange();
    assert(CFIDirectivesRange.first == CFIDirectivesRange.second &&
           "CFI directives should be in some frame");
    return;
  }

  const auto *LastFrame = &getDwarfFrameInfos()[FrameIndices.back()];

  auto DirectivesRange = updateDirectivesRange();
  ArrayRef<MCCFIInstruction> Directives;
  if (DirectivesRange.first < DirectivesRange.second) {
    Directives = ArrayRef<MCCFIInstruction>(LastFrame->Instructions);
    Directives =
        Directives.drop_front(DirectivesRange.first)
            .drop_back(LastFrame->Instructions.size() - DirectivesRange.second);
  }

  if (LastInstruction) {
    assert(!UIAs.empty() && "If the instruction is in a frame, there should be "
                            "a analysis instantiated for it");
    UIAs.back().update(LastInstruction.value(), Directives);
  } else {
    assert(!DirectivesRange.first &&
           "An analysis should be created at the begining of a frame");
    UIAs.emplace_back(&getContext(), MCII,
                      false /* TODO should put isEH here */, Directives);
  }
}

void FunctionUnitStreamer::emitInstruction(const MCInst &Inst,
                                           const MCSubtargetInfo &STI) {
  updateUIA();
  LastInstruction = Inst;
}

void FunctionUnitStreamer::emitCFIStartProcImpl(MCDwarfFrameInfo &Frame) {
  updateUIA();
  FrameIndices.push_back(getNumFrameInfos());
  LastInstruction = std::nullopt;
  LastDirectiveIndex = 0;
  MCStreamer::emitCFIStartProcImpl(Frame);
}

void FunctionUnitStreamer::emitCFIEndProcImpl(MCDwarfFrameInfo &CurFrame) {
  updateUIA();

  assert(!FrameIndices.empty() && "There should be at least one frame to pop");
  FrameIndices.pop_back();

  assert(!UIAs.empty() && "There should be an analysis for each frame");
  UIAs.pop_back();

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
