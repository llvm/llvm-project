//===-- SystemZMCAsmInfo.cpp - SystemZ asm properties ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZMCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"

using namespace llvm;

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {MCSymbolRefExpr::VK_DTPOFF, "DTPOFF"},
    {MCSymbolRefExpr::VK_GOT, "GOT"},
    {MCSymbolRefExpr::VK_GOTENT, "GOTENT"},
    {MCSymbolRefExpr::VK_INDNTPOFF, "INDNTPOFF"},
    {MCSymbolRefExpr::VK_NTPOFF, "NTPOFF"},
    {MCSymbolRefExpr::VK_PLT, "PLT"},
    {MCSymbolRefExpr::VK_TLSGD, "TLSGD"},
    {MCSymbolRefExpr::VK_TLSLD, "TLSLD"},
    {MCSymbolRefExpr::VK_TLSLDM, "TLSLDM"},
};

SystemZMCAsmInfoELF::SystemZMCAsmInfoELF(const Triple &TT) {
  AssemblerDialect = AD_GNU;
  CalleeSaveStackSlotSize = 8;
  CodePointerSize = 8;
  Data64bitsDirective = "\t.quad\t";
  ExceptionsType = ExceptionHandling::DwarfCFI;
  IsLittleEndian = false;
  MaxInstLength = 6;
  SupportsDebugInformation = true;
  UsesELFSectionDirectiveForBSS = true;
  ZeroDirective = "\t.space\t";

  initializeVariantKinds(variantKindDescs);
}

SystemZMCAsmInfoGOFF::SystemZMCAsmInfoGOFF(const Triple &TT) {
  AllowAdditionalComments = false;
  AllowAtInName = true;
  AllowAtAtStartOfIdentifier = true;
  AllowDollarAtStartOfIdentifier = true;
  AssemblerDialect = AD_HLASM;
  CalleeSaveStackSlotSize = 8;
  CodePointerSize = 8;
  CommentString = "*";
  ExceptionsType = ExceptionHandling::ZOS;
  IsHLASM = true;
  IsLittleEndian = false;
  MaxInstLength = 6;
  SupportsDebugInformation = true;

  initializeVariantKinds(variantKindDescs);
}

bool SystemZMCAsmInfoGOFF::isAcceptableChar(char C) const {
  return MCAsmInfo::isAcceptableChar(C) || C == '#';
}
