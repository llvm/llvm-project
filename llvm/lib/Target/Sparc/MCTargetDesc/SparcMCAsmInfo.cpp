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
#include "llvm/ADT/Triple.h"

using namespace llvm;

void SparcELFMCAsmInfo::anchor() { }

SparcELFMCAsmInfo::SparcELFMCAsmInfo(const Target &T, StringRef TT) {
  IsLittleEndian = false;
  Triple TheTriple(TT);
  if (TheTriple.getArch() == Triple::sparcv9)
    PointerSize = 8;

  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
  Data64bitsDirective = 0;  // .xword is only supported by V9.
  ZeroDirective = "\t.skip\t";
  CommentString = "!";
  HasLEB128 = true;
  SupportsDebugInformation = true;
  
  SunStyleELFSectionSwitchSyntax = true;
  UsesELFSectionDirectiveForBSS = true;

  WeakRefDirective = "\t.weak\t";

  PrivateGlobalPrefix = ".L";
}


