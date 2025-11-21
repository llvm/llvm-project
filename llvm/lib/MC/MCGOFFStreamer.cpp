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
#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCGOFFObjectWriter.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSymbolGOFF.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

MCGOFFStreamer::MCGOFFStreamer(MCContext &Context,
                               std::unique_ptr<MCAsmBackend> MAB,
                               std::unique_ptr<MCObjectWriter> OW,
                               std::unique_ptr<MCCodeEmitter> Emitter)
    : MCObjectStreamer(Context, std::move(MAB), std::move(OW),
                       std::move(Emitter)) {}

MCGOFFStreamer::~MCGOFFStreamer() = default;

void MCGOFFStreamer::finishImpl() {
  getWriter().setRootSD(static_cast<MCSectionGOFF *>(
                            getContext().getObjectFileInfo()->getTextSection())
                            ->getParent());
  MCObjectStreamer::finishImpl();
}

GOFFObjectWriter &MCGOFFStreamer::getWriter() {
  return static_cast<GOFFObjectWriter &>(getAssembler().getWriter());
}

void MCGOFFStreamer::changeSection(MCSection *Section, uint32_t Subsection) {
  // Make sure that all section are registered in the correct order.
  SmallVector<MCSectionGOFF *> Sections;
  for (auto *S = static_cast<MCSectionGOFF *>(Section); S; S = S->getParent())
    Sections.push_back(S);
  while (!Sections.empty()) {
    auto *S = Sections.pop_back_val();
    MCObjectStreamer::changeSection(S, Sections.empty() ? Subsection : 0);
  }
}

void MCGOFFStreamer::emitLabel(MCSymbol *Symbol, SMLoc Loc) {
  MCObjectStreamer::emitLabel(Symbol, Loc);
}

bool MCGOFFStreamer::emitSymbolAttribute(MCSymbol *Sym,
                                         MCSymbolAttr Attribute) {
  auto *Symbol = static_cast<MCSymbolGOFF *>(Sym);
  switch (Attribute) {
  case MCSA_Invalid:
  case MCSA_Cold:
  case MCSA_ELF_TypeIndFunction:
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

  case MCSA_ELF_TypeFunction:
    Symbol->setCodeData(GOFF::ESDExecutable::ESD_EXE_CODE);
    break;
  case MCSA_ELF_TypeObject:
    Symbol->setCodeData(GOFF::ESDExecutable::ESD_EXE_DATA);
    break;
  case MCSA_OSLinkage:
    Symbol->setLinkage(GOFF::ESDLinkageType::ESD_LT_OS);
    break;
  case MCSA_XPLinkage:
    Symbol->setLinkage(GOFF::ESDLinkageType::ESD_LT_XPLink);
    break;
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

void MCGOFFStreamer::emitExterns() {
}

MCStreamer *llvm::createGOFFStreamer(MCContext &Context,
                                     std::unique_ptr<MCAsmBackend> &&MAB,
                                     std::unique_ptr<MCObjectWriter> &&OW,
                                     std::unique_ptr<MCCodeEmitter> &&CE) {
  MCGOFFStreamer *S =
      new MCGOFFStreamer(Context, std::move(MAB), std::move(OW), std::move(CE));
  return S;
}
