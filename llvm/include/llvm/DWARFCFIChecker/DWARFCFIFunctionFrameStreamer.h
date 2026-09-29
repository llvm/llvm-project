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
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Compiler.h"
#include <memory>
#include <optional>

namespace llvm {

/// This class is an `MCStreamer` implementation that watches for machine
/// instructions and CFI directives. It cuts the stream into function frames and
/// channels them to `CFIFunctionFrameReceiver`. A function frame is the machine
/// instructions and CFI directives that are between `.cfi_startproc` and
/// `.cfi_endproc` directives.
class LLVM_ABI CFIFunctionFrameStreamer : public MCStreamer {
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
  /// This method sends the last instruction, along with its associated
  /// directives, to the receiver and then updates the internal state of the
  /// class. It moves the directive index to after the last directive and sets
  /// the last instruction to \p NewInst . This method assumes it is called in
  /// the middle of an unfinished DWARF debug frame; if not, an assertion will
  /// fail.
  void updateReceiver(const std::optional<MCInst> &NewInst);

private:
  /// The following fields are stacks that store the state of the stream sent to
  /// the receiver in each frame. This class, like `MCStreamer`, assumes that
  /// the debug frames are intertwined with each other only in stack form.

  /// The last instruction that is not sent to the receiver for each frame.
  SmallVector<std::optional<MCInst>> LastInstructions;
  /// The index of the last directive that is not sent to the receiver for each
  /// frame.
  SmallVector<unsigned> LastDirectiveIndices;
  /// The index of each frame in `DwarfFrameInfos` field in `MCStreamer`.
  SmallVector<unsigned> FrameIndices;

  std::unique_ptr<CFIFunctionFrameReceiver> Receiver;
};

} // namespace llvm

#endif
