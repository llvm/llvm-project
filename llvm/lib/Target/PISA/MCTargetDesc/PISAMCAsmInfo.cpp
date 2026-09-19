//===-- PISAMCAsmInfo.cpp - PISA asm properties ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISAMCAsmInfo.h"

#include "llvm/TargetParser/Triple.h"

using namespace llvm;

PISAMCAsmInfo::PISAMCAsmInfo(const Triple &TT, const MCTargetOptions &Options)
    : MCAsmInfo(Options) {
  IsLittleEndian = true;

  HasSingleParameterDotFile = false;
  HasDotTypeDotSizeDirective = false;

  MinInstAlignment = 4;
  CodePointerSize = 4;
  HasFunctionAlignment = false;

  SeparatorString = ";";
  CommentString = "//";

  InlineAsmStart = "Inline assembly start";
  InlineAsmEnd = "Inline assembly end";

  UseIntegratedAssembler = false;

  // Allow '$' in identifier names so that register names like %$ or %$foo
  // are correctly lexed as identifiers rather than as AsmToken::Dollar.
  AllowDollarAtStartOfIdentifier = true;
}

bool PISAMCAsmInfo::shouldOmitSectionDirective(StringRef SectionName) const {
  return true;
}
