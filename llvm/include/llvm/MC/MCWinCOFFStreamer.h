//===- MCWinCOFFStreamer.h - COFF Object File Interface ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCWINCOFFSTREAMER_H
#define LLVM_MC_MCWINCOFFSTREAMER_H

#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCObjectStreamer.h"

namespace llvm {

class MCAsmBackend;
class MCContext;
class MCCodeEmitter;
class MCInst;
class MCSection;
class MCSubtargetInfo;
class MCSymbol;
class StringRef;
class raw_pwrite_stream;

class MCWinCOFFStreamer : public MCObjectStreamer {
public:
  MCWinCOFFStreamer(MCContext &Context, std::unique_ptr<MCAsmBackend> MAB,
                    std::unique_ptr<MCCodeEmitter> CE,
                    std::unique_ptr<MCObjectWriter> OW);

  /// state management
  void reset() override {
    CurSymbol = nullptr;
    MCObjectStreamer::reset();
  }

  /// \name MCStreamer interface
  /// \{

  void InitSections(bool NoExecStack) override;
  void EmitLabel(MCSymbol *Symbol, SMLoc Loc = SMLoc()) override;
  void EmitAssemblerFlag(MCAssemblerFlag Flag) override;
  void EmitThumbFunc(MCSymbol *Func) override;
  bool EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override;
  void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) override;
  void BeginCOFFSymbolDef(MCSymbol const *Symbol) override;
  void EmitCOFFSymbolStorageClass(int StorageClass) override;
  void EmitCOFFSymbolType(int Type) override;
  void EndCOFFSymbolDef() override;
  void EmitCOFFSafeSEH(MCSymbol const *Symbol) override;
  void EmitCOFFSymbolIndex(MCSymbol const *Symbol) override;
  void EmitCOFFSectionIndex(MCSymbol const *Symbol) override;
  void EmitCOFFSecRel32(MCSymbol const *Symbol, uint64_t Offset) override;
  void EmitCOFFImgRel32(MCSymbol const *Symbol, int64_t Offset) override;
  void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        unsigned ByteAlignment) override;
  void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                             unsigned ByteAlignment) override;
  void EmitZerofill(MCSection *Section, MCSymbol *Symbol, uint64_t Size,
                    unsigned ByteAlignment, SMLoc Loc = SMLoc()) override;
  void EmitTBSSSymbol(MCSection *Section, MCSymbol *Symbol, uint64_t Size,
                      unsigned ByteAlignment) override;
  void EmitIdent(StringRef IdentString) override;
  void EmitWinEHHandlerData(SMLoc Loc) override;
  void FinishImpl() override;

  /// \}

protected:
  const MCSymbol *CurSymbol;

  void EmitInstToData(const MCInst &Inst, const MCSubtargetInfo &STI) override;

private:
  void Error(const Twine &Msg) const;
};

} // end namespace llvm

#endif // LLVM_MC_MCWINCOFFSTREAMER_H
