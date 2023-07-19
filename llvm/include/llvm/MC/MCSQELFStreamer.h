//===- MCSQELFStreamer.h - MCStreamer SQlite ELF Object File Interface ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSQELFSTREAMER_H
#define LLVM_MC_MCSQELFSTREAMER_H

#include "llvm/MC/MCObjectStreamer.h"

namespace llvm {

class MCContext;
class MCDataFragment;
class MCFragment;
class MCObjectWriter;
class MCSection;
class MCSubtargetInfo;
class MCSymbol;
class MCSymbolRefExpr;
class MCAsmBackend;
class MCCodeEmitter;
class MCExpr;
class MCInst;

class MCSQELFStreamer : public MCObjectStreamer {
public:
  MCSQELFStreamer(MCContext &Context, std::unique_ptr<MCAsmBackend> TAB,
                std::unique_ptr<MCObjectWriter> OW,
                std::unique_ptr<MCCodeEmitter> Emitter);

  ~MCSQELFStreamer() override = default;

  bool emitSymbolAttribute(MCSymbol *Symbol,
                           MCSymbolAttr Attribute) override;

  void emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        Align ByteAlignment) override;

  void emitZerofill(MCSection *Section, MCSymbol *Symbol = nullptr,
                    uint64_t Size = 0, Align ByteAlignment = Align(1),
                    SMLoc Loc = SMLoc()) override;       

  void emitInstToData(const MCInst &Inst, const MCSubtargetInfo&) override;        
private:

};
} // end namespace llvm

#endif // LLVM_MC_MCSQELFSTREAMER_H
