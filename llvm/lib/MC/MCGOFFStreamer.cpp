//===- lib/MC/MCGOFFStreamer.cpp - GOFF Object Output ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file assembles .s files and emits GOFF .o object files.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCGOFFStreamer.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCGOFFObjectWriter.h"
#include "llvm/MC/MCSymbolGOFF.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

MCGOFFStreamer::~MCGOFFStreamer() {}

GOFFObjectWriter &MCGOFFStreamer::getWriter() {
  return static_cast<GOFFObjectWriter &>(getAssembler().getWriter());
}

// Make sure that all section are registered in the correct order.
static void registerSectionHierarchy(MCAssembler &Asm, MCSectionGOFF *Section) {
  if (Section->isRegistered())
    return;
  if (Section->getParent())
    registerSectionHierarchy(Asm, Section->getParent());
  Asm.registerSection(*Section);
}

void MCGOFFStreamer::changeSection(MCSection *Section, uint32_t Subsection) {
  registerSectionHierarchy(getAssembler(),
                           static_cast<MCSectionGOFF *>(Section));
  MCObjectStreamer::changeSection(Section, Subsection);
}

void MCGOFFStreamer::emitLabel(MCSymbol *Symbol, SMLoc Loc) {
  MCObjectStreamer::emitLabel(Symbol, Loc);
  cast<MCSymbolGOFF>(Symbol)->initAttributes();
}

bool MCGOFFStreamer::emitSymbolAttribute(MCSymbol *Sym,
                                         MCSymbolAttr Attribute) {
  auto *Symbol = cast<MCSymbolGOFF>(Sym);
  switch (Attribute) {
  case MCSA_Invalid:
  case MCSA_Cold:
  case MCSA_ELF_TypeFunction:
  case MCSA_ELF_TypeIndFunction:
  case MCSA_ELF_TypeObject:
  case MCSA_ELF_TypeTLS:
  case MCSA_ELF_TypeCommon:
  case MCSA_ELF_TypeNoType:
  case MCSA_ELF_TypeGnuUniqueObject:
  case MCSA_LGlobal:
  case MCSA_Extern:
  case MCSA_Exported:
  case MCSA_IndirectSymbol:
  case MCSA_Internal:
  case MCSA_LazyReference:
  case MCSA_NoDeadStrip:
  case MCSA_SymbolResolver:
  case MCSA_AltEntry:
  case MCSA_PrivateExtern:
  case MCSA_Protected:
  case MCSA_Reference:
  case MCSA_WeakDefinition:
  case MCSA_WeakDefAutoPrivate:
  case MCSA_WeakAntiDep:
  case MCSA_Memtag:
    return false;

  case MCSA_Global:
    Symbol->setExternal(true);
    break;
  case MCSA_Local:
    Symbol->setExternal(false);
    break;
  case MCSA_Weak:
  case MCSA_WeakReference:
    Symbol->setExternal(true);
    Symbol->setWeak();
    break;
  case MCSA_Hidden:
    Symbol->setHidden(true);
    break;
  }

  return true;
}

MCStreamer *llvm::createGOFFStreamer(MCContext &Context,
                                     std::unique_ptr<MCAsmBackend> &&MAB,
                                     std::unique_ptr<MCObjectWriter> &&OW,
                                     std::unique_ptr<MCCodeEmitter> &&CE) {
  MCGOFFStreamer *S =
      new MCGOFFStreamer(Context, std::move(MAB), std::move(OW), std::move(CE));
  return S;
}
