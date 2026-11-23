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

constexpr EnumStringDef<MCAsmInfo::AtSpecifierKind> AtSpecifierDefsELF[] = {
    {{"DTPOFF"}, SystemZ::S_DTPOFF}, {{"GOT"}, SystemZ::S_GOT},
    {{"GOTENT"}, SystemZ::S_GOTENT}, {{"INDNTPOFF"}, SystemZ::S_INDNTPOFF},
    {{"NTPOFF"}, SystemZ::S_NTPOFF}, {{"PLT"}, SystemZ::S_PLT},
    {{"TLSGD"}, SystemZ::S_TLSGD},   {{"TLSLD"}, SystemZ::S_TLSLD},
    {{"TLSLDM"}, SystemZ::S_TLSLDM},
};
constexpr EnumStringDef<MCAsmInfo::AtSpecifierKind> AtSpecifierDefsGOFF[] = {
    {{"AD"}, SystemZ::S_None}, {{"QD"}, SystemZ::S_QCon},
    {{"RD"}, SystemZ::S_RCon}, {{"VD"}, SystemZ::S_VCon},
    {{"JD"}, SystemZ::S_JCon},
};
constexpr auto atSpecifiersELF = BUILD_ENUM_STRINGS(AtSpecifierDefsELF);
constexpr auto atSpecifiersGOFF = BUILD_ENUM_STRINGS(AtSpecifierDefsGOFF);

SystemZMCAsmInfoELF::SystemZMCAsmInfoELF(const Triple &TT,
                                         const MCTargetOptions &Options)
    : MCAsmInfoELF(Options) {
  AssemblerDialect = AD_GNU;
  CalleeSaveStackSlotSize = 8;
  CodePointerSize = 8;
  Data64bitsDirective = "\t.quad\t";
  IsLittleEndian = false;
  MaxInstLength = 6;
  SupportsDebugInformation = true;
  UsesELFSectionDirectiveForBSS = true;
  ZeroDirective = "\t.space\t";

  initializeAtSpecifiers(atSpecifiersELF);
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
  IsHLASM = true;
  IsLittleEndian = false;
  MaxInstLength = 6;
  SupportsDebugInformation = true;

  initializeAtSpecifiers(atSpecifiersGOFF);
}

const MCExpr *SystemZMCAsmInfoGOFF::getDwarfSectionOffsetExpr(
    const MCSymbol *Sym, uint64_t Offset, MCContext &Ctx) const {
  const MCExpr *Expr = MCSpecifierExpr::create(Sym, SystemZ::S_QCon, Ctx);
  if (Offset)
    Expr =
        MCBinaryExpr::createAdd(Expr, MCConstantExpr::create(Offset, Ctx), Ctx);
  return Expr;
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
  case SystemZ::S_JCon:
    OS << "JD";
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
