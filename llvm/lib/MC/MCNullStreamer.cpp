//===- lib/MC/MCNullStreamer.cpp - Dummy Streamer Implementation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCStreamer.h"
namespace llvm {
class MCContext;
class MCExpr;
class MCSection;
class MCSymbol;
} // namespace llvm

using namespace llvm;

namespace {

class MCNullStreamer : public MCStreamer {
public:
  MCNullStreamer(MCContext &Context) : MCStreamer(Context) {}

  /// @name MCStreamer Interface
  /// @{

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
  void emitXCOFFLocalCommonSymbol(MCSymbol *LabelSym, uint64_t Size,
                                  MCSymbol *CsectSym,
                                  Align Alignment) override {}
  void emitXCOFFSymbolLinkageWithVisibility(MCSymbol *Symbol,
                                            MCSymbolAttr Linkage,
                                            MCSymbolAttr Visibility) override {}
  void emitXCOFFExceptDirective(const MCSymbol *Symbol, const MCSymbol *Trap,
                                unsigned Lang, unsigned Reason,
                                unsigned FunctionSize, bool hasDebug) override {
  }
  void emitXCOFFRefDirective(const MCSymbol *Symbol) override {}
  void emitXCOFFCInfoSym(StringRef Name, StringRef Metadata) override {}
};

} // namespace

MCStreamer *llvm::createNullStreamer(MCContext &Context) {
  return new MCNullStreamer(Context);
}
