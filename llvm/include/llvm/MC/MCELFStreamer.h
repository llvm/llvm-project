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
  void ChangeSection(MCSection *Section, const MCExpr *Subsection) override;
  void EmitLabel(MCSymbol *Symbol, SMLoc Loc = SMLoc()) override;
  void EmitLabelAtPos(MCSymbol *Symbol, SMLoc Loc, MCFragment *F,
                      uint64_t Offset) override;
  void EmitAssemblerFlag(MCAssemblerFlag Flag) override;
  void EmitThumbFunc(MCSymbol *Func) override;
  void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) override;
  bool EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override;
  void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) override;
  void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        unsigned ByteAlignment) override;

  void emitELFSize(MCSymbol *Symbol, const MCExpr *Value) override;
  void emitELFSymverDirective(StringRef AliasName,
                              const MCSymbol *Aliasee) override;

  void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                             unsigned ByteAlignment) override;

  void EmitZerofill(MCSection *Section, MCSymbol *Symbol = nullptr,
                    uint64_t Size = 0, unsigned ByteAlignment = 0,
                    SMLoc L = SMLoc()) override;
  void EmitTBSSSymbol(MCSection *Section, MCSymbol *Symbol, uint64_t Size,
                      unsigned ByteAlignment = 0) override;
  void EmitValueImpl(const MCExpr *Value, unsigned Size,
                     SMLoc Loc = SMLoc()) override;

  void EmitIdent(StringRef IdentString) override;

  void EmitValueToAlignment(unsigned, int64_t, unsigned, unsigned) override;

  void emitCGProfileEntry(const MCSymbolRefExpr *From,
                          const MCSymbolRefExpr *To, uint64_t Count) override;

  void FinishImpl() override;

  void EmitBundleAlignMode(unsigned AlignPow2) override;
  void EmitBundleLock(bool AlignToEnd) override;
  void EmitBundleUnlock() override;

private:
  bool isBundleLocked() const;
  void EmitInstToFragment(const MCInst &Inst, const MCSubtargetInfo &) override;
  void EmitInstToData(const MCInst &Inst, const MCSubtargetInfo &) override;

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
