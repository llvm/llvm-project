//===- MCELFStreamer.h - MCStreamer ELF Object File Interface ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCELFSTREAMER_H
#define LLVM_MC_MCELFSTREAMER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCObjectStreamer.h"

namespace llvm {

class MCAsmBackend;
class MCCodeEmitter;
class MCExpr;
class MCInst;

class MCELFStreamer : public MCObjectStreamer {
public:
  MCELFStreamer(MCContext &Context, std::unique_ptr<MCAsmBackend> TAB,
                std::unique_ptr<MCObjectWriter> OW,
                std::unique_ptr<MCCodeEmitter> Emitter);

  ~MCELFStreamer() override = default;

  /// state management
  void reset() override {
    SeenIdent = false;
    BundleGroups.clear();
    MCObjectStreamer::reset();
  }

  /// \name MCStreamer Interface
  /// @{

  void InitSections(bool NoExecStack) override;
  void changeSection(MCSection *Section, const MCExpr *Subsection) override;
  void emitLabel(MCSymbol *Symbol, SMLoc Loc = SMLoc()) override;
  void emitLabelAtPos(MCSymbol *Symbol, SMLoc Loc, MCFragment *F,
                      uint64_t Offset) override;
  void emitAssemblerFlag(MCAssemblerFlag Flag) override;
  void emitThumbFunc(MCSymbol *Func) override;
  void emitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) override;
  bool emitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override;
  void emitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) override;
  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        unsigned ByteAlignment) override;

  void emitELFSize(MCSymbol *Symbol, const MCExpr *Value) override;
  void emitELFSymverDirective(const MCSymbol *OriginalSym, StringRef Name,
                              bool KeepOriginalSym) override;

  void emitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                             unsigned ByteAlignment) override;

  void emitZerofill(MCSection *Section, MCSymbol *Symbol = nullptr,
                    uint64_t Size = 0, unsigned ByteAlignment = 0,
                    SMLoc L = SMLoc()) override;
  void emitTBSSSymbol(MCSection *Section, MCSymbol *Symbol, uint64_t Size,
                      unsigned ByteAlignment = 0) override;
  void emitValueImpl(const MCExpr *Value, unsigned Size,
                     SMLoc Loc = SMLoc()) override;

  void emitIdent(StringRef IdentString) override;

  void emitValueToAlignment(unsigned, int64_t, unsigned, unsigned) override;

  void emitCGProfileEntry(const MCSymbolRefExpr *From,
                          const MCSymbolRefExpr *To, uint64_t Count) override;

  void finishImpl() override;

  void emitBundleAlignMode(unsigned AlignPow2) override;
  void emitBundleLock(bool AlignToEnd) override;
  void emitBundleUnlock() override;

private:
  bool isBundleLocked() const;
  void emitInstToFragment(const MCInst &Inst, const MCSubtargetInfo &) override;
  void emitInstToData(const MCInst &Inst, const MCSubtargetInfo &) override;

  void fixSymbolsInTLSFixups(const MCExpr *expr);
  void finalizeCGProfileEntry(const MCSymbolRefExpr *&S);
  void finalizeCGProfile();

  /// Merge the content of the fragment \p EF into the fragment \p DF.
  void mergeFragment(MCDataFragment *, MCDataFragment *);

  bool SeenIdent = false;

  /// BundleGroups - The stack of fragments holding the bundle-locked
  /// instructions.
  SmallVector<MCDataFragment *, 4> BundleGroups;
};

MCELFStreamer *createARMELFStreamer(MCContext &Context,
                                    std::unique_ptr<MCAsmBackend> TAB,
                                    std::unique_ptr<MCObjectWriter> OW,
                                    std::unique_ptr<MCCodeEmitter> Emitter,
                                    bool RelaxAll, bool IsThumb, bool IsAndroid);

} // end namespace llvm

#endif // LLVM_MC_MCELFSTREAMER_H
