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
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionGOFF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolGOFF.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"

#define DEBUG_TYPE "goff-streamer"

using namespace llvm;

MCGOFFStreamer::~MCGOFFStreamer() {}

MCStreamer *llvm::createGOFFStreamer(MCContext &Context,
                                     std::unique_ptr<MCAsmBackend> &&MAB,
                                     std::unique_ptr<MCObjectWriter> &&OW,
                                     std::unique_ptr<MCCodeEmitter> &&CE,
                                     bool RelaxAll) {
  MCGOFFStreamer *S =
      new MCGOFFStreamer(Context, std::move(MAB), std::move(OW), std::move(CE));
  if (RelaxAll)
    S->getAssembler().setRelaxAll(true);
  return S;
}

void MCGOFFStreamer::initSections(bool NoExecStack,
                                  const MCSubtargetInfo &STI) {
  MCContext &Ctx = getContext();
  if (NoExecStack)
    switchSection(Ctx.getAsmInfo()->getNonexecutableStackSection(Ctx));
  else
    switchSection(Ctx.getObjectFileInfo()->getTextSection());
}

void MCGOFFStreamer::switchSection(MCSection *S, const MCExpr *Subsection) {
  auto Section = cast<MCSectionGOFF>(S);
  MCSection *Parent = Section->getParent();

  if (Parent) {
    const MCExpr *Subsection = Section->getSubsectionId();
    assert(Subsection && "No subsection associated with child section");
    this->MCObjectStreamer::switchSection(Parent, Subsection);
    return;
  }

  this->MCObjectStreamer::switchSection(Section, Subsection);
}

bool MCGOFFStreamer::emitSymbolAttribute(MCSymbol *S, MCSymbolAttr Attribute) {
  auto *Symbol = cast<MCSymbolGOFF>(S);

  getAssembler().registerSymbol(*Symbol);

  switch (Attribute) {
  case MCSA_Global:
    Symbol->setExternal(true);
    break;
  case MCSA_Local:
    Symbol->setExternal(false);
    break;
  case MCSA_Hidden:
    Symbol->setHidden(true);
    break;
  case MCSA_Weak:
  case MCSA_WeakReference:
    Symbol->setExternal(true);
    Symbol->setWeak();
    break;
  case MCSA_ELF_TypeFunction:
    Symbol->setExecutable(GOFF::ESD_EXE_CODE);
    break;
  case MCSA_ELF_TypeObject:
    Symbol->setExecutable(GOFF::ESD_EXE_DATA);
    break;
  case MCSA_ZOS_OS_Linkage:
    Symbol->setOSLinkage();
    break;
  default:
    return false;
  }
  return true;
}

void MCGOFFStreamer::emitInstToData(const MCInst &Inst,
                                    const MCSubtargetInfo &STI) {
  LLVM_DEBUG(dbgs() << "Entering " << __PRETTY_FUNCTION__ << "\n");
  LLVM_DEBUG(dbgs() << "Inst: " << Inst << "\n");

  MCAssembler &Assembler = getAssembler();
  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  Assembler.getEmitter().encodeInstruction(Inst, Code, Fixups, STI);

  // Bundling is not currently supported.
  assert(!Assembler.isBundlingEnabled() && "Do not handle bundling yet");

  MCDataFragment *DF = getOrCreateDataFragment();

  // Add the fixups and data.
  for (unsigned I = 0, E = Fixups.size(); I != E; ++I) {
    Fixups[I].setOffset(Fixups[I].getOffset() + DF->getContents().size());
    DF->getFixups().push_back(Fixups[I]);
  }
  DF->setHasInstructions(STI);
  DF->getContents().append(Code.begin(), Code.end());
}
