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
#include "llvm/MC/MCValue.h"

using namespace llvm;

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {SystemZ::S_DTPOFF, "DTPOFF"}, {SystemZ::S_GOT, "GOT"},
    {SystemZ::S_GOTENT, "GOTENT"}, {SystemZ::S_INDNTPOFF, "INDNTPOFF"},
    {SystemZ::S_NTPOFF, "NTPOFF"}, {SystemZ::S_PLT, "PLT"},
    {SystemZ::S_TLSGD, "TLSGD"},   {SystemZ::S_TLSLD, "TLSLD"},
    {SystemZ::S_TLSLDM, "TLSLDM"},
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
  UsesSetToEquateSymbol = true;
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

void SystemZMCAsmInfoGOFF::printSpecifierExpr(
    raw_ostream &OS, const MCSpecifierExpr &Expr) const {
  switch (Expr.getSpecifier()) {
  case SystemZ::S_None:
    OS << "A";
    break;
  case SystemZ::S_RCon:
    OS << "R";
    break;
  case SystemZ::S_VCon:
    OS << "V";
    break;
  default:
    llvm_unreachable("Invalid kind");
  }
  OS << '(';
  printExpr(OS, *Expr.getSubExpr());
  OS << ')';
}

bool SystemZMCAsmInfoGOFF::evaluateAsRelocatableImpl(
    const MCSpecifierExpr &Expr, MCValue &Res, const MCAssembler *Asm) const {
  if (!Expr.getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;
  Res.setSpecifier(Expr.getSpecifier());
  return true;
}
