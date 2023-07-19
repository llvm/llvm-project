//===- lib/MC/MCSQELFStreamer.cpp -SQLite ELF Object Output -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file assembles .s files and emits SQLite ELF object files.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSQELFStreamer.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

MCSQELFStreamer::MCSQELFStreamer(MCContext &Context,
                             std::unique_ptr<MCAsmBackend> TAB,
                             std::unique_ptr<MCObjectWriter> OW,
                             std::unique_ptr<MCCodeEmitter> Emitter)
    : MCObjectStreamer(Context, std::move(TAB), std::move(OW),
                       std::move(Emitter)) {}


bool MCSQELFStreamer::emitSymbolAttribute(MCSymbol *Symbol,
                         MCSymbolAttr Attribute) {
  return true;
}

void MCSQELFStreamer::emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                      Align ByteAlignment) {                        
}

void MCSQELFStreamer::emitZerofill(MCSection *Section, MCSymbol *Symbol,
                  uint64_t Size, Align ByteAlignment, SMLoc Loc) {

}       

void MCSQELFStreamer::emitInstToData(const MCInst &Inst, const MCSubtargetInfo&) {

}     

MCStreamer *llvm::createSQELFStreamer(MCContext &Context,
                                    std::unique_ptr<MCAsmBackend> &&MAB,
                                    std::unique_ptr<MCObjectWriter> &&OW,
                                    std::unique_ptr<MCCodeEmitter> &&CE,
                                    bool RelaxAll) {
  MCSQELFStreamer *S =
      new MCSQELFStreamer(Context, std::move(MAB), std::move(OW), std::move(CE));
  if (RelaxAll)
    S->getAssembler().setRelaxAll(true);
  return S;
}
