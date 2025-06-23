//===- VEMCAsmInfo.cpp - VE asm properties --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the VEMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "VEMCAsmInfo.h"
#include "VEMCExpr.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCValue.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {VE::S_HI32, "hi"},
    {VE::S_LO32, "lo"},
    {VE::S_PC_HI32, "pc_hi"},
    {VE::S_PC_LO32, "pc_lo"},
    {VE::S_GOT_HI32, "got_hi"},
    {VE::S_GOT_LO32, "got_lo"},
    {VE::S_GOTOFF_HI32, "gotoff_hi"},
    {VE::S_GOTOFF_LO32, "gotoff_lo"},
    {VE::S_PLT_HI32, "plt_hi"},
    {VE::S_PLT_LO32, "plt_lo"},
    {VE::S_TLS_GD_HI32, "tls_gd_hi"},
    {VE::S_TLS_GD_LO32, "tls_gd_lo"},
    {VE::S_TPOFF_HI32, "tpoff_hi"},
    {VE::S_TPOFF_LO32, "tpoff_lo"},
};

void VEELFMCAsmInfo::anchor() {}

VEELFMCAsmInfo::VEELFMCAsmInfo(const Triple &TheTriple) {

  CodePointerSize = CalleeSaveStackSlotSize = 8;
  MaxInstLength = MinInstAlignment = 8;

  // VE uses ".*byte" directive for unaligned data.
  Data8bitsDirective = "\t.byte\t";
  Data16bitsDirective = "\t.2byte\t";
  Data32bitsDirective = "\t.4byte\t";
  Data64bitsDirective = "\t.8byte\t";

  // Uses '.section' before '.bss' directive.  VE requires this although
  // assembler manual says sinple '.bss' is supported.
  UsesELFSectionDirectiveForBSS = true;

  SupportsDebugInformation = true;

  initializeVariantKinds(variantKindDescs);
}

void VEELFMCAsmInfo::printSpecifierExpr(raw_ostream &OS,
                                        const MCSpecifierExpr &Expr) const {
  printExpr(OS, *Expr.getSubExpr());
  auto specifier = Expr.getSpecifier();
  if (specifier && specifier != VE::S_REFLONG)
    OS << '@' << getSpecifierName(specifier);
}

bool VEELFMCAsmInfo::evaluateAsRelocatableImpl(const MCSpecifierExpr &Expr,
                                               MCValue &Res,
                                               const MCAssembler *Asm) const {
  if (!Expr.getSubExpr()->evaluateAsRelocatable(Res, Asm))
    return false;
  Res.setSpecifier(Expr.getSpecifier());
  return true;
}
