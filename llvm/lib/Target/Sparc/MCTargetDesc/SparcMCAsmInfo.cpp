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
    : MCAsmInfoELF(TheTriple, Options) {
  bool isV9 = TheTriple.isSPARC64();
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

  DwarfFDERelSymbolSpec = ELF::R_SPARC_DISP32;

  SpecifierExprPrintFn = [](const MCAsmInfo &MAI, raw_ostream &OS,
                            const MCSpecifierExpr &Expr) {
    StringRef S = Sparc::getSpecifierName(Expr.getSpecifier());
    if (!S.empty())
      OS << '%' << S << '(';
    MAI.printExpr(OS, *Expr.getSubExpr());
    if (!S.empty())
      OS << ')';
  };
}
