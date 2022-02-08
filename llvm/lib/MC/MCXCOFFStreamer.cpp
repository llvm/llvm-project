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

#include "llvm/MC/MCXCOFFStreamer.h"
#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionXCOFF.h"
#include "llvm/MC/MCSymbolXCOFF.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

MCXCOFFStreamer::MCXCOFFStreamer(MCContext &Context,
                                 std::unique_ptr<MCAsmBackend> MAB,
                                 std::unique_ptr<MCObjectWriter> OW,
                                 std::unique_ptr<MCCodeEmitter> Emitter)
    : MCObjectStreamer(Context, std::move(MAB), std::move(OW),
                       std::move(Emitter)) {}

bool MCXCOFFStreamer::emitSymbolAttribute(MCSymbol *Sym,
                                          MCSymbolAttr Attribute) {
  auto *Symbol = cast<MCSymbolXCOFF>(Sym);
  getAssembler().registerSymbol(*Symbol);

  switch (Attribute) {
  case MCSA_Global:
  case MCSA_Extern:
    Symbol->setStorageClass(XCOFF::C_EXT);
    Symbol->setExternal(true);
    break;
  case MCSA_LGlobal:
    Symbol->setStorageClass(XCOFF::C_HIDEXT);
    Symbol->setExternal(true);
    break;
  case llvm::MCSA_Weak:
    Symbol->setStorageClass(XCOFF::C_WEAKEXT);
    Symbol->setExternal(true);
    break;
  case llvm::MCSA_Hidden:
    Symbol->setVisibilityType(XCOFF::SYM_V_HIDDEN);
    break;
  case llvm::MCSA_Protected:
    Symbol->setVisibilityType(XCOFF::SYM_V_PROTECTED);
    break;
  default:
    report_fatal_error("Not implemented yet.");
  }
  return true;
}

void MCXCOFFStreamer::emitXCOFFSymbolLinkageWithVisibility(
    MCSymbol *Symbol, MCSymbolAttr Linkage, MCSymbolAttr Visibility) {

  emitSymbolAttribute(Symbol, Linkage);

  // When the caller passes `MCSA_Invalid` for the visibility, do not emit one.
  if (Visibility == MCSA_Invalid)
    return;

  emitSymbolAttribute(Symbol, Visibility);
}

void MCXCOFFStreamer::emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) {
  getAssembler().registerSymbol(*Symbol);
  Symbol->setExternal(cast<MCSymbolXCOFF>(Symbol)->getStorageClass() !=
                      XCOFF::C_HIDEXT);
  Symbol->setCommon(Size, ByteAlignment);

  // Default csect align is 4, but common symbols have explicit alignment values
  // and we should honor it.
  cast<MCSymbolXCOFF>(Symbol)->getRepresentedCsect()->setAlignment(
      Align(ByteAlignment));

  // Emit the alignment and storage for the variable to the section.
  emitValueToAlignment(ByteAlignment);
  emitZeros(Size);
}

void MCXCOFFStreamer::emitZerofill(MCSection *Section, MCSymbol *Symbol,
                                   uint64_t Size, unsigned ByteAlignment,
                                   SMLoc Loc) {
  report_fatal_error("Zero fill not implemented for XCOFF.");
}

void MCXCOFFStreamer::emitInstToData(const MCInst &Inst,
                                     const MCSubtargetInfo &STI) {
  MCAssembler &Assembler = getAssembler();
  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  Assembler.getEmitter().encodeInstruction(Inst, VecOS, Fixups, STI);

  // Add the fixups and data.
  MCDataFragment *DF = getOrCreateDataFragment(&STI);
  const size_t ContentsSize = DF->getContents().size();
  auto &DataFragmentFixups = DF->getFixups();
  for (auto &Fixup : Fixups) {
    Fixup.setOffset(Fixup.getOffset() + ContentsSize);
    DataFragmentFixups.push_back(Fixup);
  }

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

void MCXCOFFStreamer::emitXCOFFLocalCommonSymbol(MCSymbol *LabelSym,
                                                 uint64_t Size,
                                                 MCSymbol *CsectSym,
                                                 unsigned ByteAlignment) {
  emitCommonSymbol(CsectSym, Size, ByteAlignment);
}
