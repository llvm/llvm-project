//==-- SystemZTargetStreamer.cpp - SystemZ Target Streamer Methods ----------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines SystemZ-specific target streamer classes.
/// These are for implementing support for target-specific assembly directives.
///
//===----------------------------------------------------------------------===//

#include "SystemZTargetStreamer.h"
#include "SystemZHLASMAsmStreamer.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"

using namespace llvm;

void SystemZTargetStreamer::emitConstantPools() {
  // Emit EXRL target instructions.
  if (EXRLTargets2Sym.empty())
    return;
  // Switch to the .text section.
  const MCObjectFileInfo &OFI = *Streamer.getContext().getObjectFileInfo();
  Streamer.switchSection(OFI.getTextSection());
  for (auto &I : EXRLTargets2Sym) {
    Streamer.emitLabel(I.second);
    const MCInstSTIPair &MCI_STI = I.first;
    Streamer.emitInstruction(MCI_STI.first, *MCI_STI.second);
  }
  EXRLTargets2Sym.clear();
}

SystemZHLASMAsmStreamer &SystemZTargetHLASMStreamer::getHLASMStreamer() {
  return static_cast<SystemZHLASMAsmStreamer &>(getStreamer());
}

void SystemZTargetHLASMStreamer::emitExtern(StringRef Sym) {
  getStreamer().emitRawText(Twine(" EXTRN ") + Twine(Sym));
}

void SystemZTargetHLASMStreamer::emitEnd() { getHLASMStreamer().emitEnd(); }

// HLASM statements can only perform a single operation at a time
const MCExpr *SystemZTargetHLASMStreamer::createWordDiffExpr(
    MCContext &Ctx, const MCSymbol *Hi, const MCSymbol *Lo) {
  assert(Hi && Lo && "Symbols required to calculate expression");
  MCSymbol *Temp = Ctx.createTempSymbol();
  OS << Temp->getName() << " EQU ";
  const MCBinaryExpr *TempExpr = MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(Hi, Ctx), MCSymbolRefExpr::create(Lo, Ctx), Ctx);
  Ctx.getAsmInfo()->printExpr(OS, *TempExpr);
  OS << "\n";
  return MCBinaryExpr::createLShr(MCSymbolRefExpr::create(Temp, Ctx),
                                  MCConstantExpr::create(1, Ctx), Ctx);
}

const MCExpr *SystemZTargetGOFFStreamer::createWordDiffExpr(
    MCContext &Ctx, const MCSymbol *Hi, const MCSymbol *Lo) {
  assert(Hi && Lo && "Symbols required to calculate expression");
  return MCBinaryExpr::createLShr(
      MCBinaryExpr::createSub(MCSymbolRefExpr::create(Hi, Ctx),
                              MCSymbolRefExpr::create(Lo, Ctx), Ctx),
      MCConstantExpr::create(1, Ctx), Ctx);
}
