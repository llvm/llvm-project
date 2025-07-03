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

#ifndef LLVM_DWARFCFICHECKER_DWARFCFIFUNCTIONFRAMESTREAMER_H
#define LLVM_DWARFCFICHECKER_DWARFCFIFUNCTIONFRAMESTREAMER_H

#include "DWARFCFIFunctionFrameReceiver.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include <memory>
#include <optional>

namespace llvm {

/// This class is an `MCStreamer` implementation that watches for machine
/// instructions and CFI directives. It cuts the stream into function frames and
/// channels them to `CFIFunctionFrameReceiver`. A function frame is the machine
/// instructions and CFI directives that are between `.cfi_startproc` and
/// `.cfi_endproc` directives.
class CFIFunctionFrameStreamer : public MCStreamer {
public:
  CFIFunctionFrameStreamer(MCContext &Context,
                           std::unique_ptr<CFIFunctionFrameReceiver> Receiver)
      : MCStreamer(Context), Receiver(std::move(Receiver)) {
    assert(this->Receiver && "Receiver should not be null");
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
  void updateReceiver(const std::optional<MCInst> &NewInst);

private:
  std::vector<std::optional<MCInst>> FrameLastInstructions; //! FIXME
  std::vector<unsigned> FrameLastDirectiveIndices;          //! FIXME
  std::vector<unsigned> FrameIndices;
  std::unique_ptr<CFIFunctionFrameReceiver> Receiver;
};

} // namespace llvm

#endif
