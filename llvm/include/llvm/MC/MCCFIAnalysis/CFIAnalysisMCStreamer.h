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

    CFIDirectivesState();

    CFIDirectivesState(int FrameIndex, int InstructionIndex);
  } LastCFIDirectivesState;
  std::vector<int> FrameIndices;
  std::vector<CFIAnalysis> CFIAs;

  struct ICFI {
    MCInst Instruction;
    std::pair<unsigned, unsigned> CFIDirectivesRange;

    ICFI(MCInst Instruction, std::pair<unsigned, unsigned> CFIDirectives);
  };

  std::optional<MCInst> LastInstruction;

  std::pair<unsigned, unsigned> getCFIDirectivesRange();

  void feedCFIA();

public:
  CFIAnalysisMCStreamer(MCContext &Context, const MCInstrInfo &MCII,
                        std::unique_ptr<MCInstrAnalysis> MCIA);

  bool hasRawTextSupport() const override;
  void emitRawTextImpl(StringRef String) override;
  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override;
  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        Align ByteAlignment) override;
  void beginCOFFSymbolDef(const MCSymbol *Symbol) override;
  void emitCOFFSymbolStorageClass(int StorageClass) override;
  void emitCOFFSymbolType(int Type) override;
  void endCOFFSymbolDef() override;
  void emitXCOFFSymbolLinkageWithVisibility(MCSymbol *Symbol,
                                            MCSymbolAttr Linkage,
                                            MCSymbolAttr Visibility) override;
  void emitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override;
  void emitCFIStartProcImpl(MCDwarfFrameInfo &Frame) override;
  void emitCFIEndProcImpl(MCDwarfFrameInfo &CurFrame) override;
};

} // namespace llvm
#endif