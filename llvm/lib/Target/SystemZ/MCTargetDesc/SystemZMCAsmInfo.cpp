//===-- SystemZMCAsmInfo.cpp - SystemZ asm properties ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZMCAsmInfo.h"
#include "MCTargetDesc/SystemZMCExpr.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"

using namespace llvm;

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {SystemZMCExpr::VK_DTPOFF, "DTPOFF"},
    {SystemZMCExpr::VK_GOT, "GOT"},
    {SystemZMCExpr::VK_GOTENT, "GOTENT"},
    {SystemZMCExpr::VK_INDNTPOFF, "INDNTPOFF"},
    {SystemZMCExpr::VK_NTPOFF, "NTPOFF"},
    {SystemZMCExpr::VK_PLT, "PLT"},
    {SystemZMCExpr::VK_TLSGD, "TLSGD"},
    {SystemZMCExpr::VK_TLSLD, "TLSLD"},
    {SystemZMCExpr::VK_TLSLDM, "TLSLDM"},
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
