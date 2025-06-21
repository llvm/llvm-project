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
#include "MCTargetDesc/HexagonMCExpr.h"
#include "llvm/MC/MCExpr.h"

using namespace llvm;

const MCAsmInfo::VariantKindDesc variantKindDescs[] = {
    {HexagonMCExpr::VK_DTPREL, "DTPREL"}, {HexagonMCExpr::VK_GD_GOT, "GDGOT"},
    {HexagonMCExpr::VK_GD_PLT, "GDPLT"},  {HexagonMCExpr::VK_GOT, "GOT"},
    {HexagonMCExpr::VK_GOTREL, "GOTREL"}, {HexagonMCExpr::VK_IE, "IE"},
    {HexagonMCExpr::VK_IE_GOT, "IEGOT"},  {HexagonMCExpr::VK_LD_GOT, "LDGOT"},
    {HexagonMCExpr::VK_LD_PLT, "LDPLT"},  {HexagonMCExpr::VK_PCREL, "PCREL"},
    {HexagonMCExpr::VK_PLT, "PLT"},       {HexagonMCExpr::VK_TPREL, "TPREL"},
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
  UsesSetToEquateSymbol = true;
  ZeroDirective = "\t.space\t";
  AscizDirective = "\t.string\t";

  MinInstAlignment = 4;
  UsesELFSectionDirectiveForBSS  = true;
  ExceptionsType = ExceptionHandling::DwarfCFI;
  UseLogicalShr = false;

  initializeVariantKinds(variantKindDescs);
}
