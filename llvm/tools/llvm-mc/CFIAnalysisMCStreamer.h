#ifndef LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_MC_STREAMER_H
#define LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_MC_STREAMER_H

#include "CFIAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
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
    int FrameIndex; // TODO remove it, no need for it
    int DirectiveIndex;

    CFIDirectivesState() : FrameIndex(-1), DirectiveIndex(0) {}

    CFIDirectivesState(int FrameIndex, int InstructionIndex)
        : FrameIndex(FrameIndex), DirectiveIndex(InstructionIndex) {}
  } LastCFIDirectivesState;
  std::vector<int> FrameIndices;
  std::vector<CFIAnalysis> CFIAs;

  struct ICFI {
    MCInst Instruction;
    ArrayRef<MCCFIInstruction> CFIDirectives;

    ICFI(MCInst Instruction, ArrayRef<MCCFIInstruction> CFIDirectives)
        : Instruction(Instruction), CFIDirectives(CFIDirectives) {}
  };

  std::optional<MCInst> LastInstruction;
  std::optional<MCDwarfFrameInfo> LastDwarfFrameInfo;

  std::optional<ICFI> getLastICFI() {
    if (!LastInstruction)
      return std::nullopt;

    auto DwarfFrameInfos = getDwarfFrameInfos();
    int FrameIndex = FrameIndices.back();
    auto CurrentCFIDirectiveState =
        hasUnfinishedDwarfFrameInfo()
            ? CFIDirectivesState(
                  FrameIndex, DwarfFrameInfos[FrameIndex].Instructions.size())
            : CFIDirectivesState();
    assert(CurrentCFIDirectiveState.FrameIndex ==
           LastCFIDirectivesState.FrameIndex);
    assert(CurrentCFIDirectiveState.DirectiveIndex >=
           LastCFIDirectivesState.DirectiveIndex);

    std::vector<MCCFIInstruction> CFIDirectives;
    if (LastCFIDirectivesState.FrameIndex >= 0) {
      auto CFIInstructions = DwarfFrameInfos[FrameIndex].Instructions;
      CFIDirectives = std::vector<MCCFIInstruction>(
          CFIInstructions.begin() + LastCFIDirectivesState.DirectiveIndex,
          CFIInstructions.begin() + CurrentCFIDirectiveState.DirectiveIndex);
    }

    LastCFIDirectivesState = CurrentCFIDirectiveState;
    return ICFI(LastInstruction.value(), CFIDirectives);
  }

  void feedCFIA() {
    if (!LastDwarfFrameInfo) {
      // TODO Maybe this corner case causes bugs, when the programmer did a
      // mistake in the startproc, endprocs and also made a mistake in not
      // adding cfi directives for a instruction. Then this would cause to
      // ignore the instruction.
      auto LastICFI = getLastICFI();
      assert(!LastICFI || LastICFI->CFIDirectives.empty());
      return;
    }

    if (auto ICFI = getLastICFI()) {
      assert(!CFIAs.empty());
      CFIAs.back().update(LastDwarfFrameInfo.value(), ICFI->Instruction,
                          ICFI->CFIDirectives);
    }
  }

public:
  CFIAnalysisMCStreamer(MCContext &Context, const MCInstrInfo &MCII,
                        std::unique_ptr<MCInstrAnalysis> MCIA)
      : MCStreamer(Context), MCII(MCII), MCIA(std::move(MCIA)),
        LastCFIDirectivesState(), LastInstruction(std::nullopt) {
    FrameIndices.push_back(-1);
  }

  /// @name MCStreamer Interface
  /// @{

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
    if (hasUnfinishedDwarfFrameInfo())
      LastDwarfFrameInfo = getDwarfFrameInfos()[FrameIndices.back()];
    else
      LastDwarfFrameInfo = std::nullopt;
  }

  void emitCFIStartProcImpl(MCDwarfFrameInfo &Frame) override {
    feedCFIA();
    FrameIndices.push_back(getNumFrameInfos());
    CFIAs.emplace_back(getContext(), MCII, MCIA.get());
    LastInstruction = std::nullopt;
    LastDwarfFrameInfo = std::nullopt;
    LastCFIDirectivesState.FrameIndex = FrameIndices.back();
    LastCFIDirectivesState.DirectiveIndex = 0;
    MCStreamer::emitCFIStartProcImpl(Frame);
  }

  void emitCFIEndProcImpl(MCDwarfFrameInfo &CurFrame) override {
    feedCFIA();
    // TODO this will break if the input frame are malformed.
    FrameIndices.pop_back();
    CFIAs.pop_back();
    LastInstruction = std::nullopt;
    LastDwarfFrameInfo = std::nullopt;
    LastCFIDirectivesState.FrameIndex = FrameIndices.back();
    LastCFIDirectivesState.DirectiveIndex =
        LastCFIDirectivesState.FrameIndex >= 0
            ? getDwarfFrameInfos()[LastCFIDirectivesState.FrameIndex]
                  .Instructions.size()
            : 0;
    MCStreamer::emitCFIEndProcImpl(CurFrame);
  }
};

} // namespace llvm
#endif