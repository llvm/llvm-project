//===- MCELFStreamer.h - MCStreamer ELF Object File Interface ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
  MCELFStreamer(MCContext &Context, MCAsmBackend &TAB, raw_pwrite_stream &OS,
                MCCodeEmitter *Emitter)
      : MCObjectStreamer(Context, TAB, OS, Emitter) {}

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
  void EmitAssemblerFlag(MCAssemblerFlag Flag) override;
  void EmitThumbFunc(MCSymbol *Func) override;
  void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) override;
  bool EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override;
  void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) override;
  void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        unsigned ByteAlignment) override;

  void emitELFSize(MCSymbol *Symbol, const MCExpr *Value) override;

  void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                             unsigned ByteAlignment) override;

  void EmitZerofill(MCSection *Section, MCSymbol *Symbol = nullptr,
                    uint64_t Size = 0, unsigned ByteAlignment = 0) override;
  void EmitTBSSSymbol(MCSection *Section, MCSymbol *Symbol, uint64_t Size,
                      unsigned ByteAlignment = 0) override;
  void EmitValueImpl(const MCExpr *Value, unsigned Size,
                     SMLoc Loc = SMLoc()) override;

  void EmitIdent(StringRef IdentString) override;

  void EmitValueToAlignment(unsigned, int64_t, unsigned, unsigned) override;

  void FinishImpl() override;

  void EmitBundleAlignMode(unsigned AlignPow2) override;
  void EmitBundleLock(bool AlignToEnd) override;
  void EmitBundleUnlock() override;

private:
  bool isBundleLocked() const;
  void EmitInstToFragment(const MCInst &Inst, const MCSubtargetInfo &) override;
  void EmitInstToData(const MCInst &Inst, const MCSubtargetInfo &) override;

  void fixSymbolsInTLSFixups(const MCExpr *expr);

  /// \brief Merge the content of the fragment \p EF into the fragment \p DF.
  void mergeFragment(MCDataFragment *, MCDataFragment *);

  bool SeenIdent = false;

  /// BundleGroups - The stack of fragments holding the bundle-locked
  /// instructions.
  SmallVector<MCDataFragment *, 4> BundleGroups;
};

MCELFStreamer *createARMELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                                    raw_pwrite_stream &OS,
                                    MCCodeEmitter *Emitter, bool RelaxAll,
                                    bool IsThumb);

} // end namespace llvm

#endif // LLVM_MC_MCELFSTREAMER_H
