//===-- SystemZMCAsmInfo.cpp - SystemZ asm properties ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZMCAsmInfo.h"
#include "llvm/ADT/Enum.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCValue.h"

using namespace llvm;

constexpr EnumStringDef<MCAsmInfo::AtSpecifierKind> AtSpecifierDefs[] = {
    {{"DTPOFF"}, SystemZ::S_DTPOFF}, {{"GOT"}, SystemZ::S_GOT},
    {{"GOTENT"}, SystemZ::S_GOTENT}, {{"INDNTPOFF"}, SystemZ::S_INDNTPOFF},
    {{"NTPOFF"}, SystemZ::S_NTPOFF}, {{"PLT"}, SystemZ::S_PLT},
    {{"TLSGD"}, SystemZ::S_TLSGD},   {{"TLSLD"}, SystemZ::S_TLSLD},
    {{"TLSLDM"}, SystemZ::S_TLSLDM},
};
constexpr auto atSpecifiers = BUILD_ENUM_STRINGS(AtSpecifierDefs);

SystemZMCAsmInfoELF::SystemZMCAsmInfoELF(const Triple &TT,
                                         const MCTargetOptions &Options)
    : MCAsmInfoELF(Options) {
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

  initializeAtSpecifiers(atSpecifiers);
}

SystemZMCAsmInfoGOFF::SystemZMCAsmInfoGOFF(const Triple &TT,
                                           const MCTargetOptions &Options)
    : MCAsmInfoGOFF(Options) {
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

  initializeAtSpecifiers(atSpecifiers);
}

void SystemZMCAsmInfoGOFF::printSpecifierExpr(
    raw_ostream &OS, const MCSpecifierExpr &Expr) const {
  switch (Expr.getSpecifier()) {
  case SystemZ::S_None:
    OS << "AD";
    break;
  case SystemZ::S_QCon:
    OS << "QD";
    break;
  case SystemZ::S_RCon:
    OS << "RD";
    break;
  case SystemZ::S_VCon:
    OS << "VD";
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
