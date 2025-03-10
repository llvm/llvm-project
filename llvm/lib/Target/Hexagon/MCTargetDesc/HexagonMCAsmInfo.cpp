//===-- HexagonMCAsmInfo.cpp - Hexagon asm properties ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the HexagonMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "HexagonMCAsmInfo.h"
#include "llvm/MC/MCExpr.h"

using namespace llvm;

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {MCSymbolRefExpr::VK_DTPREL, "DTPREL"},
    {MCSymbolRefExpr::VK_Hexagon_GD_GOT, "GDGOT"},
    {MCSymbolRefExpr::VK_Hexagon_GD_PLT, "GDPLT"},
    {MCSymbolRefExpr::VK_GOT, "GOT"},
    {MCSymbolRefExpr::VK_GOTREL, "GOTREL"},
    {MCSymbolRefExpr::VK_Hexagon_IE, "IE"},
    {MCSymbolRefExpr::VK_Hexagon_IE_GOT, "IEGOT"},
    {MCSymbolRefExpr::VK_Hexagon_LD_GOT, "LDGOT"},
    {MCSymbolRefExpr::VK_Hexagon_LD_PLT, "LDPLT"},
    {MCSymbolRefExpr::VK_PCREL, "PCREL"},
    {MCSymbolRefExpr::VK_PLT, "PLT"},
    {MCSymbolRefExpr::VK_TPREL, "TPREL"},
};

// Pin the vtable to this file.
void HexagonMCAsmInfo::anchor() {}

HexagonMCAsmInfo::HexagonMCAsmInfo(const Triple &TT) {
  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
  Data64bitsDirective = nullptr;  // .xword is only supported by V9.
  CommentString = "//";
  SupportsDebugInformation = true;

  LCOMMDirectiveAlignmentType = LCOMM::ByteAlignment;
  InlineAsmStart = "# InlineAsm Start";
  InlineAsmEnd = "# InlineAsm End";
  ZeroDirective = "\t.space\t";
  AscizDirective = "\t.string\t";

  MinInstAlignment = 4;
  UsesELFSectionDirectiveForBSS  = true;
  ExceptionsType = ExceptionHandling::DwarfCFI;
  UseLogicalShr = false;

  initializeVariantKinds(variantKindDescs);
}
