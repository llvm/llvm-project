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
#include "llvm/MC/MCGOFFObjectWriter.h"
#include "llvm/MC/TargetRegistry.h"

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

MCStreamer *llvm::createGOFFStreamer(MCContext &Context,
                                     std::unique_ptr<MCAsmBackend> &&MAB,
                                     std::unique_ptr<MCObjectWriter> &&OW,
                                     std::unique_ptr<MCCodeEmitter> &&CE) {
  MCGOFFStreamer *S =
      new MCGOFFStreamer(Context, std::move(MAB), std::move(OW), std::move(CE));
  return S;
}
