//===- MCWasmStreamer.h - MCStreamer Wasm Object File Interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCWASMSTREAMER_H
#define LLVM_MC_MCWASMSTREAMER_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCAsmBackend;
class MCAssembler;
class MCCodeEmitter;
class MCExpr;
class MCInst;
class raw_ostream;

class MCWasmStreamer : public MCObjectStreamer {
public:
  MCWasmStreamer(MCContext &Context, MCAsmBackend &TAB, raw_pwrite_stream &OS,
                 MCCodeEmitter *Emitter)
      : MCObjectStreamer(Context, TAB, OS, Emitter), SeenIdent(false) {}

  ~MCWasmStreamer() override;

  /// state management
  void reset() override {
    SeenIdent = false;
    MCObjectStreamer::reset();
  }

  /// \name MCStreamer Interface
  /// @{

  void ChangeSection(MCSection *Section, const MCExpr *Subsection) override;
  void EmitAssemblerFlag(MCAssemblerFlag Flag) override;
  void EmitThumbFunc(MCSymbol *Func) override;
  void EmitWeakReference(MCSymbol *Alias, const MCSymbol *Symbol) override;
  bool EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override;
  void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) override;
  void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        unsigned ByteAlignment) override;
  void BeginCOFFSymbolDef(const MCSymbol *Symbol) override;
  void EmitCOFFSymbolStorageClass(int StorageClass) override;
  void EmitCOFFSymbolType(int Type) override;
  void EndCOFFSymbolDef() override;

  void emitELFSize(MCSymbol *Symbol, const MCExpr *Value) override;

  void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                             unsigned ByteAlignment) override;

  void EmitZerofill(MCSection *Section, MCSymbol *Symbol = nullptr,
                    uint64_t Size = 0, unsigned ByteAlignment = 0) override;
  void EmitTBSSSymbol(MCSection *Section, MCSymbol *Symbol, uint64_t Size,
                      unsigned ByteAlignment = 0) override;
  void EmitValueImpl(const MCExpr *Value, unsigned Size,
                     SMLoc Loc = SMLoc()) override;

  void EmitFileDirective(StringRef Filename) override;

  void EmitIdent(StringRef IdentString) override;

  void EmitValueToAlignment(unsigned, int64_t, unsigned, unsigned) override;

  void FinishImpl() override;

private:
  void EmitInstToFragment(const MCInst &Inst, const MCSubtargetInfo &) override;
  void EmitInstToData(const MCInst &Inst, const MCSubtargetInfo &) override;

  /// \brief Merge the content of the fragment \p EF into the fragment \p DF.
  void mergeFragment(MCDataFragment *, MCDataFragment *);

  bool SeenIdent;
};

} // end namespace llvm

#endif
