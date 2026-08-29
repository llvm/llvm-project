//===- SparcMCAsmInfo.cpp - Sparc asm properties --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SparcMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SparcMCAsmInfo.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

void SparcELFMCAsmInfo::anchor() {}

SparcELFMCAsmInfo::SparcELFMCAsmInfo(const Triple &TheTriple,
                                     const MCTargetOptions &Options)
    : MCAsmInfoELF(Options) {
  bool isV9 = (TheTriple.getArch() == Triple::sparcv9);
  IsLittleEndian = (TheTriple.getArch() == Triple::sparcel);

  if (isV9) {
    CodePointerSize = CalleeSaveStackSlotSize = 8;
  }

  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
  // .xword is only supported by V9.
  Data64bitsDirective = (isV9) ? "\t.xword\t" : nullptr;
  ZeroDirective = "\t.skip\t";
  CommentString = "!";
  SupportsDebugInformation = true;

  ExceptionsType = ExceptionHandling::DwarfCFI;

  UsesELFSectionDirectiveForBSS = true;

  DwarfFDERelSymbolSpec = ELF::R_SPARC_DISP32;
}

void SparcELFMCAsmInfo::printSpecifierExpr(raw_ostream &OS,
                                           const MCSpecifierExpr &Expr) const {
  StringRef S = Sparc::getSpecifierName(Expr.getSpecifier());
  if (!S.empty())
    OS << '%' << S << '(';
  printExpr(OS, *Expr.getSubExpr());
  if (!S.empty())
    OS << ')';
}
