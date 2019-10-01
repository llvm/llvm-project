//===- MC/MCAsmInfoXCOFF.cpp - XCOFF asm properties ------------ *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfoXCOFF.h"

using namespace llvm;

void MCAsmInfoXCOFF::anchor() {}

MCAsmInfoXCOFF::MCAsmInfoXCOFF() {
  IsLittleEndian = false;
  HasDotTypeDotSizeDirective = false;
  COMMDirectiveAlignmentIsInBytes = false;
  LCOMMDirectiveAlignmentType = LCOMM::Log2Alignment;
  UseDotAlignForAlignment = true;
  AsciiDirective = nullptr; // not supported
  AscizDirective = nullptr; // not supported
  NeedsFunctionDescriptors = true;
  HasDotLGloblDirective = true;
  Data64bitsDirective = "\t.llong\t";
  SupportsQuotedNames = false;
}

bool MCAsmInfoXCOFF::isValidUnquotedName(StringRef Name) const {
  // FIXME: Remove this function when we stop using "TOC[TC0]" as a symbol name.
  if (Name.equals("TOC[TC0]"))
    return true;

  return MCAsmInfo::isValidUnquotedName(Name);
}
