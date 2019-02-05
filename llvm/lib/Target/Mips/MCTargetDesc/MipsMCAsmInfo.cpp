//===-- MipsMCAsmInfo.cpp - Mips Asm Properties ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the MipsMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "MipsMCAsmInfo.h"
#include "llvm/ADT/Triple.h"

using namespace llvm;

void MipsMCAsmInfo::anchor() { }

MipsMCAsmInfo::MipsMCAsmInfo(const Triple &TheTriple) {
  IsLittleEndian = TheTriple.isLittleEndian();

  if (TheTriple.isMIPS64() && TheTriple.getEnvironment() != Triple::GNUABIN32)
    CodePointerSize = CalleeSaveStackSlotSize = 8;

  // FIXME: This condition isn't quite right but it's the best we can do until
  //        this object can identify the ABI. It will misbehave when using O32
  //        on a mips64*-* triple.
  if (TheTriple.isMIPS32()) {
    PrivateGlobalPrefix = "$";
    PrivateLabelPrefix = "$";
  }

  AlignmentIsInBytes          = false;
  Data16bitsDirective         = "\t.2byte\t";
  Data32bitsDirective         = "\t.4byte\t";
  Data64bitsDirective         = "\t.8byte\t";
  CommentString               = "#";
  ZeroDirective               = "\t.space\t";
  GPRel32Directive            = "\t.gpword\t";
  GPRel64Directive            = "\t.gpdword\t";
  DTPRel32Directive           = "\t.dtprelword\t";
  DTPRel64Directive           = "\t.dtpreldword\t";
  TPRel32Directive            = "\t.tprelword\t";
  TPRel64Directive            = "\t.tpreldword\t";
  UseAssignmentForEHBegin = true;
  SupportsDebugInformation = true;
  ExceptionsType = ExceptionHandling::DwarfCFI;
  DwarfRegNumForCFI = true;
  HasMipsExpressions = true;
  UseIntegratedAssembler = true;
}
