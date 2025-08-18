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
#include "llvm/MC/MCXCOFFObjectWriter.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

MCXCOFFStreamer::MCXCOFFStreamer(MCContext &Context,
                                 std::unique_ptr<MCAsmBackend> MAB,
                                 std::unique_ptr<MCObjectWriter> OW,
                                 std::unique_ptr<MCCodeEmitter> Emitter)
    : MCObjectStreamer(Context, std::move(MAB), std::move(OW),
                       std::move(Emitter)) {}

XCOFFObjectWriter &MCXCOFFStreamer::getWriter() {
  return static_cast<XCOFFObjectWriter &>(getAssembler().getWriter());
}

void MCXCOFFStreamer::changeSection(MCSection *Section, uint32_t Subsection) {
  MCObjectStreamer::changeSection(Section, Subsection);
  auto *Sec = static_cast<const MCSectionXCOFF *>(Section);
  // We might miss calculating the symbols difference as absolute value before
  // adding fixups when symbol_A without the fragment set is the csect itself
  // and symbol_B is in it.
  // TODO: Currently we only set the fragment for XMC_PR csects and DWARF
  // sections because we don't have other cases that hit this problem yet.
  // if (IsDwarfSec || CsectProp->MappingClass == XCOFF::XMC_PR)
  //   QualName->setFragment(F);
  if (Sec->isDwarfSect() || Sec->getMappingClass() == XCOFF::XMC_PR)
    Sec->getQualNameSymbol()->setFragment(CurFrag);
}

bool MCXCOFFStreamer::emitSymbolAttribute(MCSymbol *Sym,
                                          MCSymbolAttr Attribute) {
  auto *Symbol = static_cast<MCSymbolXCOFF *>(Sym);
  getAssembler().registerSymbol(*Symbol);

  switch (Attribute) {
  // XCOFF doesn't support the cold feature.
  case MCSA_Cold:
    return false;

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
  case llvm::MCSA_Exported:
    Symbol->setVisibilityType(XCOFF::SYM_V_EXPORTED);
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

void MCXCOFFStreamer::emitXCOFFRefDirective(const MCSymbol *Symbol) {
  // Add a Fixup here to later record a relocation of type R_REF to prevent the
  // ref symbol from being garbage collected (by the binder).
  addFixup(MCSymbolRefExpr::create(Symbol, getContext()),
           XCOFF::RelocationType::R_REF);
}

void MCXCOFFStreamer::emitXCOFFRenameDirective(const MCSymbol *Name,
                                               StringRef Rename) {
  auto *Symbol = static_cast<const MCSymbolXCOFF *>(Name);
  if (!Symbol->hasRename())
    report_fatal_error("Only explicit .rename is supported for XCOFF.");
}

void MCXCOFFStreamer::emitXCOFFExceptDirective(const MCSymbol *Symbol,
                                               const MCSymbol *Trap,
                                               unsigned Lang, unsigned Reason,
                                               unsigned FunctionSize,
                                               bool hasDebug) {
  getWriter().addExceptionEntry(Symbol, Trap, Lang, Reason, FunctionSize,
                                hasDebug);
}

void MCXCOFFStreamer::emitXCOFFCInfoSym(StringRef Name, StringRef Metadata) {
  getWriter().addCInfoSymEntry(Name, Metadata);
}

void MCXCOFFStreamer::emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       Align ByteAlignment) {
  auto *Sym = static_cast<MCSymbolXCOFF *>(Symbol);
  getAssembler().registerSymbol(*Symbol);
  Symbol->setExternal(Sym->getStorageClass() != XCOFF::C_HIDEXT);
  Symbol->setCommon(Size, ByteAlignment);

  // Default csect align is 4, but common symbols have explicit alignment values
  // and we should honor it.
  Sym->getRepresentedCsect()->setAlignment(ByteAlignment);

  // Emit the alignment and storage for the variable to the section.
  emitValueToAlignment(ByteAlignment);
  emitZeros(Size);
}

void MCXCOFFStreamer::emitXCOFFLocalCommonSymbol(MCSymbol *LabelSym,
                                                 uint64_t Size,
                                                 MCSymbol *CsectSym,
                                                 Align Alignment) {
  emitCommonSymbol(CsectSym, Size, Alignment);
}
