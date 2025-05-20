#ifndef LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_MC_STREAMER_H
#define LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_MC_STREAMER_H

#include "CFIAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include <cstdio>
#include <memory>
#include <optional>

namespace llvm {

class CFIAnalysisMCStreamer : public MCStreamer {
  MCInstrInfo const &MCII;
  std::unique_ptr<MCInstrAnalysis> MCIA;

  struct CFIDirectivesState {
    int DirectiveIndex;

    CFIDirectivesState() : DirectiveIndex(0) {}

    CFIDirectivesState(int FrameIndex, int InstructionIndex)
        : DirectiveIndex(InstructionIndex) {}
  } LastCFIDirectivesState;
  std::vector<int> FrameIndices;
  std::vector<CFIAnalysis> CFIAs;

  struct ICFI {
    MCInst Instruction;
    std::pair<unsigned, unsigned> CFIDirectivesRange;

    ICFI(MCInst Instruction, std::pair<unsigned, unsigned> CFIDirectives)
        : Instruction(Instruction), CFIDirectivesRange(CFIDirectives) {}
  };

  std::optional<MCInst> LastInstruction;

  std::pair<unsigned, unsigned> getCFIDirectivesRange() {
    auto DwarfFrameInfos = getDwarfFrameInfos();
    int FrameIndex = FrameIndices.back();
    auto CurrentCFIDirectiveState =
        hasUnfinishedDwarfFrameInfo()
            ? CFIDirectivesState(
                  FrameIndex, DwarfFrameInfos[FrameIndex].Instructions.size())
            : CFIDirectivesState();
    assert(CurrentCFIDirectiveState.DirectiveIndex >=
           LastCFIDirectivesState.DirectiveIndex);

    std::pair<unsigned, unsigned> CFIDirectivesRange(
        LastCFIDirectivesState.DirectiveIndex,
        CurrentCFIDirectiveState.DirectiveIndex);
    LastCFIDirectivesState = CurrentCFIDirectiveState;
    return CFIDirectivesRange;
  }

  void feedCFIA() {
    auto FrameIndex = FrameIndices.back();
    if (FrameIndex < 0) {
      // TODO Maybe this corner case causes bugs, when the programmer did a
      // mistake in the startproc, endprocs and also made a mistake in not
      // adding cfi directives for a instruction. Then this would cause to
      // ignore the instruction.
      auto CFIDirectivesRange = getCFIDirectivesRange();
      assert(!LastInstruction ||
             CFIDirectivesRange.first == CFIDirectivesRange.second);
      return;
    }

    // TODO get this from emit yourself, instead of getting it in this way
    const auto *LastDwarfFrameInfo = &getDwarfFrameInfos()[FrameIndex];

    auto CFIDirectivesRange = getCFIDirectivesRange();
    ArrayRef<MCCFIInstruction> CFIDirectives;
    if (CFIDirectivesRange.first < CFIDirectivesRange.second) {
      CFIDirectives =
          ArrayRef<MCCFIInstruction>(LastDwarfFrameInfo->Instructions);
      CFIDirectives = CFIDirectives.drop_front(CFIDirectivesRange.first)
                          .drop_back(LastDwarfFrameInfo->Instructions.size() -
                                     CFIDirectivesRange.second);
    }

    if (LastInstruction != std::nullopt) {
      assert(!CFIAs.empty());

      CFIAs.back().update(LastInstruction.value(), CFIDirectives);
    } else {
      CFIAs.emplace_back(getContext(), MCII, MCIA.get(), CFIDirectives);
    }
  }

public:
  CFIAnalysisMCStreamer(MCContext &Context, const MCInstrInfo &MCII,
                        std::unique_ptr<MCInstrAnalysis> MCIA)
      : MCStreamer(Context), MCII(MCII), MCIA(std::move(MCIA)),
        LastCFIDirectivesState(), LastInstruction(std::nullopt) {
    FrameIndices.push_back(-1);
  }

  bool hasRawTextSupport() const override { return true; }
  void emitRawTextImpl(StringRef String) override {}

  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override {
    return true;
  }

  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        Align ByteAlignment) override {}
  void beginCOFFSymbolDef(const MCSymbol *Symbol) override {}
  void emitCOFFSymbolStorageClass(int StorageClass) override {}
  void emitCOFFSymbolType(int Type) override {}
  void endCOFFSymbolDef() override {}
  void emitXCOFFSymbolLinkageWithVisibility(MCSymbol *Symbol,
                                            MCSymbolAttr Linkage,
                                            MCSymbolAttr Visibility) override {}

  void emitInstruction(const MCInst &Inst,
                       const MCSubtargetInfo &STI) override {
    feedCFIA();
    LastInstruction = Inst;
  }

  void emitCFIStartProcImpl(MCDwarfFrameInfo &Frame) override {
    feedCFIA();
    FrameIndices.push_back(getNumFrameInfos());
    LastInstruction = std::nullopt;
    LastCFIDirectivesState.DirectiveIndex = 0;
    MCStreamer::emitCFIStartProcImpl(Frame);
  }

  void emitCFIEndProcImpl(MCDwarfFrameInfo &CurFrame) override {
    feedCFIA();
    // TODO this will break if the input frame are malformed.
    FrameIndices.pop_back();
    CFIAs.pop_back();
    LastInstruction = std::nullopt;
    auto FrameIndex = FrameIndices.back();
    LastCFIDirectivesState.DirectiveIndex =
        FrameIndex >= 0 ? getDwarfFrameInfos()[FrameIndex].Instructions.size()
                        : 0;
    MCStreamer::emitCFIEndProcImpl(CurFrame);
  }
};

} // namespace llvm
#endif