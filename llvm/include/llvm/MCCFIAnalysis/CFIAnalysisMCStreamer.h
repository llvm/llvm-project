#ifndef LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_MC_STREAMER_H
#define LLVM_TOOLS_LLVM_MC_CFI_ANALYSIS_MC_STREAMER_H

#include "UnwindInfoAnalysis.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include <cstdio>
#include <optional>

namespace llvm {

// TODO Now the design is that the streamer, make instances of the analysis and
// TODO run the analysis on the instruction, but instead the streamer should
// TODO divide the program into function units, and then emit the function units
// TODO one by one. Also in each function unit, an instruction with associated
// TODO directives should be emitted. The analysis should be run by the
// TODO receiver.
class FunctionUnitStreamer : public MCStreamer {
  MCInstrInfo const &MCII;

  std::vector<unsigned> FrameIndices;
  std::vector<UnwindInfoAnalysis> UIAs;

  unsigned LastDirectiveIndex;
  std::optional<MCInst> LastInstruction;

  void updateUIA();

  std::pair<unsigned, unsigned> updateDirectivesRange();

public:
  FunctionUnitStreamer(MCContext &Context, const MCInstrInfo &MCII)
      : MCStreamer(Context), MCII(MCII), LastDirectiveIndex(0),
        LastInstruction(std::nullopt) {}

  bool hasRawTextSupport() const override { return true; }
  void emitRawTextImpl(StringRef String) override {}

  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override {
    return true;
  }

  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        Align ByteAlignment) override {}
  void emitSubsectionsViaSymbols() override {};
  void beginCOFFSymbolDef(const MCSymbol *Symbol) override {}
  void emitCOFFSymbolStorageClass(int StorageClass) override {}
  void emitCOFFSymbolType(int Type) override {}
  void endCOFFSymbolDef() override {}
  void emitXCOFFSymbolLinkageWithVisibility(MCSymbol *Symbol,
                                            MCSymbolAttr Linkage,
                                            MCSymbolAttr Visibility) override {}

  void emitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override;
  void emitCFIStartProcImpl(MCDwarfFrameInfo &Frame) override;
  void emitCFIEndProcImpl(MCDwarfFrameInfo &CurFrame) override;
};

} // namespace llvm
#endif