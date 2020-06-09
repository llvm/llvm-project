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
  HasVisibilityOnlyWithLinkage = true;
  PrivateGlobalPrefix = "L..";
  PrivateLabelPrefix = "L..";
  SupportsQuotedNames = false;
  UseDotAlignForAlignment = true;
  ZeroDirective = "\t.space\t";
  ZeroDirectiveSupportsNonZeroValue = false;
  AsciiDirective = nullptr; // not supported
  AscizDirective = nullptr; // not supported

  // Use .vbyte for data definition to avoid directives that apply an implicit
  // alignment.
  Data16bitsDirective = "\t.vbyte\t2, ";
  Data32bitsDirective = "\t.vbyte\t4, ";

  COMMDirectiveAlignmentIsInBytes = false;
  LCOMMDirectiveAlignmentType = LCOMM::Log2Alignment;
  HasDotTypeDotSizeDirective = false;
  HasDotExternDirective = true;
  HasDotLGloblDirective = true;
  SymbolsHaveSMC = true;
  UseIntegratedAssembler = false;
  NeedsFunctionDescriptors = true;
}

bool MCAsmInfoXCOFF::isAcceptableChar(char C) const {
  // QualName is allowed for a MCSymbolXCOFF, and
  // QualName contains '[' and ']'.
  if (C == '[' || C == ']')
    return true;

  return MCAsmInfo::isAcceptableChar(C);
}
