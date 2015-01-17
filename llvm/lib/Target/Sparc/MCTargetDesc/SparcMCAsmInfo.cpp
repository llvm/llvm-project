//===-- SparcMCAsmInfo.cpp - Sparc asm properties -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SparcMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SparcMCAsmInfo.h"
#include "SparcMCExpr.h"
#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

void SparcELFMCAsmInfo::anchor() { }

SparcELFMCAsmInfo::SparcELFMCAsmInfo(StringRef TT) {
  IsLittleEndian = false;
  Triple TheTriple(TT);
  bool isV9 = (TheTriple.getArch() == Triple::sparcv9);

  if (isV9) {
    PointerSize = CalleeSaveStackSlotSize = 8;
  }

  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
  // .xword is only supported by V9.
  Data64bitsDirective = (isV9) ? "\t.xword\t" : nullptr;
  ZeroDirective = "\t.skip\t";
  CommentString = "!";
  SupportsDebugInformation = true;

  ExceptionsType = ExceptionHandling::DwarfCFI;

  SunStyleELFSectionSwitchSyntax = true;
  UsesELFSectionDirectiveForBSS = true;

  if (TheTriple.isOSSolaris() || TheTriple.isOSOpenBSD())
    UseIntegratedAssembler = true;
}

const MCExpr*
SparcELFMCAsmInfo::getExprForPersonalitySymbol(const MCSymbol *Sym,
                                               unsigned Encoding,
                                               MCStreamer &Streamer) const {
  if (Encoding & dwarf::DW_EH_PE_pcrel) {
    MCContext &Ctx = Streamer.getContext();
    return SparcMCExpr::Create(SparcMCExpr::VK_Sparc_R_DISP32,
                               MCSymbolRefExpr::Create(Sym, Ctx), Ctx);
  }

  return MCAsmInfo::getExprForPersonalitySymbol(Sym, Encoding, Streamer);
}

const MCExpr*
SparcELFMCAsmInfo::getExprForFDESymbol(const MCSymbol *Sym,
                                       unsigned Encoding,
                                       MCStreamer &Streamer) const {
  if (Encoding & dwarf::DW_EH_PE_pcrel) {
    MCContext &Ctx = Streamer.getContext();
    return SparcMCExpr::Create(SparcMCExpr::VK_Sparc_R_DISP32,
                               MCSymbolRefExpr::Create(Sym, Ctx), Ctx);
  }
  return MCAsmInfo::getExprForFDESymbol(Sym, Encoding, Streamer);
}
