//===-- Nios2MCAsmInfo.cpp - Nios2 Asm Properties -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the Nios2MCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "Nios2MCAsmInfo.h"

#include "llvm/ADT/Triple.h"

using namespace llvm;

void Nios2MCAsmInfo::anchor() {}

Nios2MCAsmInfo::Nios2MCAsmInfo(const Triple &TheTriple) {
  if ((TheTriple.getArch() == Triple::nios2))
    IsLittleEndian = true; // the default of IsLittleEndian is true

  AlignmentIsInBytes = false;
  Data16bitsDirective = "\t.2byte\t";
  Data32bitsDirective = "\t.4byte\t";
  Data64bitsDirective = "\t.8byte\t";
  PrivateLabelPrefix = ".LC";
  CommentString = "#";
  ZeroDirective = "\t.space\t";
  GPRel32Directive = "\t.gpword\t";
  GPRel64Directive = "\t.gpdword\t";
  WeakRefDirective = "\t.weak\t";
  GlobalDirective = "\t.global\t";
  AscizDirective = "\t.string\t";
  UseAssignmentForEHBegin = true;

  SupportsDebugInformation = true;
  ExceptionsType = ExceptionHandling::DwarfCFI;
  DwarfRegNumForCFI = true;
  UsesELFSectionDirectiveForBSS = true;
}
