//===- lib/MC/MCXCOFFStreamer.cpp - XCOFF Object Output -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file assembles .s files and emits XCOFF .o object files.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSymbolXCOFF.h"
#include "llvm/MC/MCXCOFFStreamer.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

MCXCOFFStreamer::MCXCOFFStreamer(MCContext &Context,
                                 std::unique_ptr<MCAsmBackend> MAB,
                                 std::unique_ptr<MCObjectWriter> OW,
                                 std::unique_ptr<MCCodeEmitter> Emitter)
    : MCObjectStreamer(Context, std::move(MAB), std::move(OW),
                       std::move(Emitter)) {}

bool MCXCOFFStreamer::EmitSymbolAttribute(MCSymbol *Sym,
                                          MCSymbolAttr Attribute) {
  auto *Symbol = cast<MCSymbolXCOFF>(Sym);
  getAssembler().registerSymbol(*Symbol);

  switch (Attribute) {
  case MCSA_Global:
    Symbol->setStorageClass(XCOFF::C_EXT);
    Symbol->setExternal(true);
    break;
  default:
    report_fatal_error("Not implemented yet.");
  }
  return true;
}

void MCXCOFFStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) {
  getAssembler().registerSymbol(*Symbol);
  Symbol->setExternal(cast<MCSymbolXCOFF>(Symbol)->getStorageClass() !=
                      XCOFF::C_HIDEXT);
  Symbol->setCommon(Size, ByteAlignment);

  // Emit the alignment and storage for the variable to the section.
  EmitValueToAlignment(ByteAlignment);
  EmitZeros(Size);
}

void MCXCOFFStreamer::EmitZerofill(MCSection *Section, MCSymbol *Symbol,
                                   uint64_t Size, unsigned ByteAlignment,
                                   SMLoc Loc) {
  report_fatal_error("Zero fill not implemented for XCOFF.");
}

void MCXCOFFStreamer::EmitInstToData(const MCInst &Inst,
                                     const MCSubtargetInfo &STI) {
  MCAssembler &Assembler = getAssembler();
  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  Assembler.getEmitter().encodeInstruction(Inst, VecOS, Fixups, STI);

  // TODO: Handle Fixups later

  MCDataFragment *DF = getOrCreateDataFragment(&STI);
  DF->setHasInstructions(STI);
  DF->getContents().append(Code.begin(), Code.end());
}

MCStreamer *llvm::createXCOFFStreamer(MCContext &Context,
                                      std::unique_ptr<MCAsmBackend> &&MAB,
                                      std::unique_ptr<MCObjectWriter> &&OW,
                                      std::unique_ptr<MCCodeEmitter> &&CE,
                                      bool RelaxAll) {
  MCXCOFFStreamer *S = new MCXCOFFStreamer(Context, std::move(MAB),
                                           std::move(OW), std::move(CE));
  if (RelaxAll)
    S->getAssembler().setRelaxAll(true);
  return S;
}

void MCXCOFFStreamer::EmitXCOFFLocalCommonSymbol(MCSymbol *LabelSym,
                                                 uint64_t Size,
                                                 MCSymbol *CsectSym,
                                                 unsigned ByteAlignment) {
  EmitCommonSymbol(CsectSym, Size, ByteAlignment);
}
