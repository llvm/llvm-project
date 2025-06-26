//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares CFIFunctionFrameStreamer class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNWINDINFOCHECKER_DWARFCFIFUNCTIONFRAMESTREAMER_H
#define LLVM_UNWINDINFOCHECKER_DWARFCFIFUNCTIONFRAMESTREAMER_H

#include "FunctionUnitAnalyzer.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include <memory>
#include <optional>

namespace llvm {

class CFIFunctionFrameStreamer : public MCStreamer {
private:
  std::pair<unsigned, unsigned> updateDirectivesRange();
  void updateAnalyzer();

public:
  CFIFunctionFrameStreamer(MCContext &Context,
                           std::unique_ptr<CFIFunctionFrameReceiver> Analyzer)
      : MCStreamer(Context), LastInstruction(std::nullopt),
        Analyzer(std::move(Analyzer)), LastDirectiveIndex(0) {
    assert(this->Analyzer && "Analyzer should not be null");
  }

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

private:
  std::vector<unsigned> FrameIndices;
  std::optional<MCInst> LastInstruction;
  std::unique_ptr<CFIFunctionFrameReceiver> Analyzer;
  unsigned LastDirectiveIndex;
};

} // namespace llvm
#endif
