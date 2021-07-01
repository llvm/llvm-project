//===-- SystemZMCAsmInfo.cpp - SystemZ asm properties ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZMCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"

using namespace llvm;

SystemZMCAsmInfo::SystemZMCAsmInfo(const Triple &TT) {
  CodePointerSize = 8;
  CalleeSaveStackSlotSize = 8;
  IsLittleEndian = false;

  AssemblerDialect = TT.isOSzOS() ? AD_HLASM : AD_ATT;

  MaxInstLength = 6;

  CommentString = AssemblerDialect == AD_HLASM ? "*" : "#";
  RestrictCommentStringToStartOfStatement = (AssemblerDialect == AD_HLASM);
  AllowAdditionalComments = (AssemblerDialect == AD_ATT);
  AllowAtAtStartOfIdentifier = (AssemblerDialect == AD_HLASM);
  AllowDollarAtStartOfIdentifier = (AssemblerDialect == AD_HLASM);
  AllowHashAtStartOfIdentifier = (AssemblerDialect == AD_HLASM);
  DotIsPC = (AssemblerDialect == AD_ATT);
  StarIsPC = (AssemblerDialect == AD_HLASM);
  EmitGNUAsmStartIndentationMarker = (AssemblerDialect == AD_ATT);
  AllowAtInName = (AssemblerDialect == AD_HLASM);
  EmitLabelsInUpperCase = (AssemblerDialect == AD_HLASM);

  ZeroDirective = "\t.space\t";
  Data64bitsDirective = "\t.quad\t";
  UsesELFSectionDirectiveForBSS = true;
  SupportsDebugInformation = true;
  ExceptionsType = ExceptionHandling::DwarfCFI;
}

bool SystemZMCAsmInfo::isAcceptableChar(char C) const {
  if (AssemblerDialect == AD_ATT)
    return MCAsmInfo::isAcceptableChar(C);

  return MCAsmInfo::isAcceptableChar(C) || C == '#';
}
