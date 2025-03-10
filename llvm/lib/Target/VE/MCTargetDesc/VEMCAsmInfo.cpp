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
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {MCSymbolRefExpr::VK_VE_HI32, "hi"},
    {MCSymbolRefExpr::VK_VE_LO32, "lo"},
    {MCSymbolRefExpr::VK_VE_PC_HI32, "pc_hi"},
    {MCSymbolRefExpr::VK_VE_PC_LO32, "pc_lo"},
    {MCSymbolRefExpr::VK_VE_GOT_HI32, "got_hi"},
    {MCSymbolRefExpr::VK_VE_GOT_LO32, "got_lo"},
    {MCSymbolRefExpr::VK_VE_GOTOFF_HI32, "gotoff_hi"},
    {MCSymbolRefExpr::VK_VE_GOTOFF_LO32, "gotoff_lo"},
    {MCSymbolRefExpr::VK_VE_PLT_HI32, "plt_hi"},
    {MCSymbolRefExpr::VK_VE_PLT_LO32, "plt_lo"},
    {MCSymbolRefExpr::VK_VE_TLS_GD_HI32, "tls_gd_hi"},
    {MCSymbolRefExpr::VK_VE_TLS_GD_LO32, "tls_gd_lo"},
    {MCSymbolRefExpr::VK_VE_TPOFF_HI32, "tpoff_hi"},
    {MCSymbolRefExpr::VK_VE_TPOFF_LO32, "tpoff_lo"},
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
